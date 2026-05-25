/*   CONFIG   */
const CSV_PATH = "./main.csv";
const TARGET_COLUMN = "Coding_Efficiency";
const TEST_RATIO = 0.2;
const EPOCHS = 120;
const BATCH_SIZE = 32;

/*   LOAD CSV   */
async function loadCSV(path) {
    const response = await fetch(path);
    const csvText = await response.text();

    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                resolve(results.data);
            }
        });
    });
}

/*   PREPARE DATA   */
function prepareData(rows, targetColumn) {
    const columns = Object.keys(rows[0]);
    const featureColumns = columns.filter(c => c !== targetColumn);
    const features = rows.map(row =>
        featureColumns.map(col => Number(row[col]))
    );
    const labels = rows.map(row => [Number(row[targetColumn])]);
    return {features,labels,featureColumns};
}

/*   NORMALIZATION   */
function normalizeTensor(tensor) {
    const min = tensor.min(0);
    const max = tensor.max(0);
    const normalized = tensor.sub(min).div(max.sub(min));
    return {normalized,min,max};
}

function denormalize(valueTensor, min, max) {
    return valueTensor.mul(max.sub(min)).add(min);
}

/*   MODEL   */
function createModel(inputShape) {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [inputShape],units: 64,activation: "relu"}));
    model.add(tf.layers.dense({units: 32,activation: "relu"}));
    model.add(tf.layers.dense({units: 16,activation: "relu"}));
    model.add(tf.layers.dense({units: 1}));
    model.compile({optimizer: tf.train.adam(0.001),loss: "meanSquaredError",metrics: ["mae"]});
    return model;
}

/*   TRAIN   */
async function train(ts,ac,dph,ey) {
    rows = await loadCSV(CSV_PATH);
    rows = rows.map(element => ({Typing_Speed_WPM: element.Typing_Speed_WPM, Typing_Accuracy_Percent: element.Typing_Accuracy_Percent, Daily_Coding_Hours:element.Daily_Coding_Hours, Problems_Solved_Per_Day: element.Problems_Solved_Per_Day, Coding_Efficiency:element.Coding_Efficiency_Score }))
    const {features,labels,featureColumns} = prepareData(rows, TARGET_COLUMN);
    console.log("Feature columns:", featureColumns);

    /*   SHUFFLE   */
    const combined = features.map((x, i) => ({x,y: labels[i]}));
    tf.util.shuffle(combined);
    const shuffledFeatures = combined.map(v => v.x);
    const shuffledLabels = combined.map(v => v.y);

    /*   TENSORS   */
    const featureTensor = tf.tensor2d(shuffledFeatures);
    const labelTensor = tf.tensor2d(shuffledLabels);

    /*   NORMALIZE   */
    const {normalized: normalizedFeatures,min: featureMin,max: featureMax} = normalizeTensor(featureTensor);
    const {normalized: normalizedLabels,min: labelMin,max: labelMax} = normalizeTensor(labelTensor);

    /*   TRAIN / TEST SPLIT   */
    const total = normalizedFeatures.shape[0];
    const testSize = Math.floor(total * TEST_RATIO);
    const trainSize = total - testSize;

    const xTrain = normalizedFeatures.slice([0, 0],[trainSize, -1]);
    const yTrain = normalizedLabels.slice([0, 0],[trainSize, -1]);
    const xTest = normalizedFeatures.slice([trainSize, 0],[testSize, -1]);
    const yTest = normalizedLabels.slice([trainSize, 0],[testSize, -1]);

    /*   CREATE MODEL   */
    const model = createModel(featureColumns.length);
    model.summary();

    /*   TRAIN MODEL   */
    await model.fit(xTrain, yTrain, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        validationSplit: 0.1,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(
                    `Epoch ${epoch + 1}`,
                    "loss =", logs.loss.toFixed(5),
                    "mae =", logs.mae.toFixed(5),
                    "val_loss =", logs.val_loss.toFixed(5)
                );
            }
        }
    });

    /*   EVALUATE   */
    const evalResult = model.evaluate(xTest, yTest);
    const loss = await evalResult[0].data();
    const mae = await evalResult[1].data();
    console.log("Test Loss:", loss[0]);
    console.log("Test MAE:", mae[0]);

    /*   SAMPLE PREDICTION   */
    // IMPORTANT:
    // must follow same order as featureColumns

    // typingSpeed,accuracy,dailyPracticeHours,experienceYears
    // const sampleInput = [72, 95,4,7];
    const sampleInput = [ts, ac,dph,ey];
    
    const inputTensor = tf.tensor2d([sampleInput]);
    const normalizedInput = inputTensor.sub(featureMin).div(featureMax.sub(featureMin));
    const prediction = model.predict(normalizedInput);
    const denormalizedPrediction = denormalize(prediction,labelMin,labelMax);
    const predictedValue =(await denormalizedPrediction.data())[0];

    console.log(`Predicted ${TARGET_COLUMN}:`,predictedValue);
    return predictedValue;
}

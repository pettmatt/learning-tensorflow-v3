import * as tf from "@tensorflow/tfjs-node" 
import { fetchTitanicData } from "./fetchData.js"
import createTfModel from "./createModel.js"

async function trainModel() {
    const [trainingDataset, testDataset] = await fetchTitanicData()
    console.log("\n============ Datasets fetched ============\n")
    console.log("Testing dataset", testDataset.shape[0])
    console.log("Training dataset", trainingDataset.shape[0])

    // Create X tensor that contains enough variables to indicate if the person survived, excluding the result
    const trainX = trainingDataset.iloc({ columns: ["1:"] }).tensor
    // Contains the answer if the person survived.
    const trainY = trainingDataset["Survived"].tensor

    const testX = testDataset.iloc({ columns: ["1:"] }).tensor
    const testY = testDataset["Survived"].tensor

    const model = createTfModel(tf, [trainX.shape[1]])

    await model.fit(trainX, trainY, {
        batchSize: 32,
        epochs: 50,
        validationData: [testX, testY],
        validationSplit: 0.2
    })
}

trainModel()
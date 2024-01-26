import * as tf from "@tensorflow/tfjs-node"
import { trainModel, prepareData } from "./training.ts"

async function main() {
    const inputShape = [12, 12, 1]
    const epochs = 20
    const testSplit = 0.05

    const generatedData = prepareData(testSplit)
    const model: tf.Sequential = await trainModel(generatedData, inputShape, epochs)

    createModel(model, generatedData)
}

async function evaluateResults(model: tf.Sequential, data: tf.Tensor<tf.Rank>[]) {
    const result = model.evaluate(data[2], data[3])
    console.log("Test Loss", result[0].dataSync())
    console.log("Test Accuracy", result[1].dataSync())
    tf.dispose(result)
}

async function createModel(model: tf.Sequential, data: tf.Tensor<tf.Rank>[]) {
    evaluateResults(model, data)
    model.save("file://./dice-model")
    process.exit()
}

main()

import * as tf from "@tensorflow/tfjs-node"
import { folderToTensors } from "./sourceHelpers"
import getModel from "./model"

async function main() {
    const [x, y] = await folderToTensors()
  
    const model = getModel()
  
    let best = 0
    // Train
    await model.fit(x, y, {
      batchSize: 256,
      validationSplit: 0.1,
      epochs: 20,
      shuffle: true,
    })
  
    // Cleanup!
    tf.dispose([x, y, model])
    console.log('Tensors in memory', tf.memory().numTensors)
}
  
main()

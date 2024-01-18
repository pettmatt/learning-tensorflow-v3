import * as tf from "@tensorflow/tfjs-node"
import { folderToTensors, saveProgression } from "./sourceHelpers.js"
import getModel from "./model.js"

async function main() {
  const [x, y] = await folderToTensors()
  const model = getModel()
  let best = 0

  await model.fit(x, y, {
    batchSize: 256,
    validationSplit: 0.1,
    epochs: 30,
    shuffle: true,
    callbacks: {
        onEpochEnd: async (_epoch, logs) => {
            if (logs.val_acc > best) {
                saveProgression(logs.val_acc)
                model.save("file://model-result/sorting_hat")
                best = logs.val_acc
            }
        }
    }
  })

  tf.dispose([x, y, model])
  console.log("Tensors in memory", tf.memory().numTensors)
}

main()

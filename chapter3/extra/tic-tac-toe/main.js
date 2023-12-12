import * as tf from "@tensorflow/tfjs"

const gc = document.querySelector("#game-container")

// O = 1
// X = -1
// Empty = 0

const boardState = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, 0],
]

const tensor = tf.tensor2d(boardState, null, "int32")

gc.innerHTML = `<p>${ JSON.stringify(tensor) }</p>`

console.log("Checking before tidy/dispose:", tf.memory().numTensors, tf.memory().numBytes)

tensor.dispose()

console.log("Checking after manual cleaning:", tf.memory().numTensors, tf.memory().numBytes)
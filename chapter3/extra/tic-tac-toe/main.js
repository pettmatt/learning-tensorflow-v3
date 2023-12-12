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

const result = tf.tensor2d(boardState, null, "int32")

gc.innerHTML = `<p>${ JSON.stringify(result) }</p>`
console.log("TF result", result)

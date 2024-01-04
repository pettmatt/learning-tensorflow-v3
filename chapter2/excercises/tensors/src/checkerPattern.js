import * as tf from "@tensorflow/tfjs"

const canvas01 = document.querySelector("#canvas01")
const canvas02 = document.querySelector("#canvas02")
const canvas03 = document.querySelector("#canvas03")
const canvas04 = document.querySelector("#canvas04")

const checkerPattern = tf.tensor([
    [0, 1, 0],
    [1, 0, 1]
])

// Bareboned without tweaks
tf.browser.toPixels(checkerPattern, canvas01)

// Repeat pattern for set amount of times
const repeat50x50 = checkerPattern.tile([50, 50])
// rescaled50x50.print()
tf.browser.toPixels(repeat50x50, canvas02)

// Scale the image to represent, what the pattern should look like
const rescaledSize = [50, 50]
const batch = tf.expandDims(checkerPattern.asType("float32"))
batch.print()
const scale50x50 = tf.image.resizeNearestNeighbor(batch, rescaledSize)
scale50x50.print()
tf.browser.toPixels(scale50x50, canvas03)

// Flip the pattern
const flipped = tf.reverse(scale50x50, 1)
tf.browser.toPixels(flipped, canvas04)

checkerPattern.print()
checkerPattern.dispose()

if (tf.memory().numTensors > 0)
    console.log("Should be cleaned", tf.memory().numTensors)
else
    console.log("There's no tensors to clean! Great!", tf.memory().numTensors)
import * as tf from "@tensorflow/tfjs"

const img = document.querySelector("#render") as HTMLImageElement
const checkerCanvas = document.querySelector("#checker") as HTMLCanvasElement
const randomCanvas = document.querySelector("#staticImage") as HTMLCanvasElement

const tree = new Image()
tree.crossOrigin = "anonymous"
tree.src = "/tree.png"

tree.onload = () => {
  const treeTensor = tf.browser.fromPixels(tree)
  console.log(`Image ${treeTensor.shape}`)
  treeTensor.dispose()
}

const lil = tf.tensor([
  [[1], [0]],
  [[0], [1]],
])

const big = lil.tile([100, 100, 1])
const pixels = big.expandDims(0)

// tf.browser.toPixels(pixels, checkerCanvas).then(() => {
//   big.dispose()
//   lil.dispose()
// })

const randomStaticValues = tf.randomUniform<tf.Rank.R3>([400, 400, 3], 0, 255, "int32")

tf.browser.toPixels(randomStaticValues, randomCanvas).then(() => {
  randomStaticValues.dispose()
  console.log("Make sure we cleaned up", tf.memory().numTensors)
})

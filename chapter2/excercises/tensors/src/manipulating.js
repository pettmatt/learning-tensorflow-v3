import * as tf from "@tensorflow/tfjs"
console.log("Manipulation script file prints")

const tensor = tf.tensor([1, 2])
console.log("Original Tensorflow tensor.", `Tensor rank ${tensor.rankType}.`, tensor.shape)

console.log("Is this a scalar?", `Tensor rank ${tensor.rankType}.`)
tensor.print() // Lvl 1

console.log("==== NEXT SECTION ====")


const t2lvl2 = tensor.expandDims()

console.log("Is this a vector?", `Tensor rank ${t2lvl2.rankType}.`)
t2lvl2.print() // Lvl 2

console.log("==== NEXT SECTION ====")


const t3lvl3 = t2lvl2.expandDims().tile([2, 2, 1])
console.log("Matrix.", `Tensor rank ${t3lvl3.rankType}.`)
t3lvl3.print() // Lvl 3

console.log("==== NEXT SECTION ====")

const t4lvl4 = t3lvl3.expandDims()
console.log(`Tensor rank ${t4lvl4.rankType}.`)
t4lvl4.print() // Lvl 4

console.log("==== NEXT SECTION ====")

const backTot3 = t4lvl4.reshape([-1, t4lvl4.shape[2], t4lvl4.shape[3]])
console.log(`Back to lvl 3. Tensor rank ${backTot3.rankType}.`)
backTot3.print() // Lvl 3

// Cleaning
tensor.dispose()
t2lvl2.dispose()
t3lvl3.dispose()
t4lvl4.dispose()
backTot3.dispose()

if (tf.memory().numTensors > 0)
    console.log("Should be cleaned", tf.memory().numTensors)
else
    console.log("There's no tensors to clean!")
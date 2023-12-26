import * as tf from "@tensorflow/tfjs"

console.log("start", tf.memory().numTensors)

const test = tf.tensor([8367677, 4209111, 4209111, 8367677, 8367677])

console.log("Contains duplicates", test.arraySync())
const {values, indices} = tf.unique(test)
console.log("Values, indices:", values, indices)
console.log("Should not contain duplicates", values.arraySync())

tf.dispose([test, values, indices])
console.log("End of the process, memory should be wiped", tf.memory().numTensors)

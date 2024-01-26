import * as tf from "@tensorflow/tfjs-node"

export async function combos(tensorArray: tf.Tensor2D[]) {
    const startSize = tensorArray.length

    for (let i = 0; i < startSize - 1; i++) {
        for(let j = i + 1; j < startSize; j++) {
            const overlay = tf.tidy(() => {
                return tf.where(
                    tf.less(tensorArray[i], tensorArray[j]),
                    tensorArray[i],
                    tensorArray[j]
                )
            })

            tensorArray.push(overlay)
        }
    }

    return tensorArray
}

export async function shuffleCombo(array01: any[], array02: any[]) {
    let counter = array01.length
    console.assert(array01.length === array02.length)
    let temp01: any, temp02: any
    let index: number = 0

    while (counter > 0) {
        // Randomize what index should be changed
        index = (Math.random() * counter) | 0
        counter--

        // Swap last elements
        temp01 = array01[counter]
        temp02 = array02[counter]
        array01[counter] = array01[index]
        array02[counter] = array02[index]
        array01[index] = temp01
        array02[index] = temp02
    }
}

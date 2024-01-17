import { default as glob } from "glob"
import tf, { TensorLike } from "@tensorflow/tfjs-node"
import fs from "fs"

export async function folderToTensors() {

}

function filesToTensors(files) {
    let ys = []
    let xs = []

    files.forEach((file) => {
        const images = fs.readFileSync(file)
        const answer = encodeDirectory(file)
        const imageTensor = tf.node.decodeImage(images, 1)

        ys.push(answer)
        xs.push(imageTensor)
    })

    return normalizeImageData(xs, ys)
}

// Because we're in ML environment, it's better to refer to a group as a number.
// In this case where we are matching image with a label, this is a valid way doing it.
function encodeDirectory(filePath: string): number {
    if (filePath.includes("owl")) return 0
    if (filePath.includes("bird")) return 1
    if (filePath.includes("lion")) return 2
    if (filePath.includes("snail")) return 3
    if (filePath.includes("snake")) return 4
    if (filePath.includes("skull")) return 5
    if (filePath.includes("tiger")) return 6
    if (filePath.includes("parrot")) return 7
    if (filePath.includes("raccoon")) return 8
    if (filePath.includes("squirrel")) return 9

    console.error("Unrecognized folder")
    process.exit(1)
}

// interface TensorDatasetObject {
//     xs: Tensor4D,
//     ys: Tensor4D,
// }

function normalizeImageData(xs: TensorLike[], ys: TensorLike) {
    console.log("============ Normalizing image data ============")
    console.log("Stacking")
    const x = tf.stack(xs)
    const y = tf.oneHot(ys, 10)

    console.log("All images have been converted to tensors:")
    console.log("x", x.shape)
    console.log("y", y.shape)

    console.log("Finally, normalizing x values")
    const xNormalized = x.div(255)
    tf.dispose([x])

    return { xs: xNormalized, ys: y }
}
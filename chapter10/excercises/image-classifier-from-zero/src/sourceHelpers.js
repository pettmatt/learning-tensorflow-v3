import { default as glob } from "glob"
import tf from "@tensorflow/tfjs-node"
import fs from "fs"

export function folderToTensors() {
    return new Promise((resolve, reject) => {
        console.log("============ Reading PNG files ============")

        glob("files/**/*.png", (error, files) => {
            if (error) {
                console.error("Failed to read PNG file(s)", error)
                console.error("Read file(s):", files)
                reject()
                process.exit(1)
            }

            const [ys, xs] = filesToTensors(files)
            // Shuffeling data is important when creating models, because allowing specific order
            // makes the model learn the pattern of the data and the order of the data.
            shuffleTensors(xs, ys)
            const [normalizedX, y] = normalizeImageData(xs, ys)
            tf.dispose([xs, ys])

            resolve([normalizedX, y])
        })
    })
}

function filesToTensors(files) {
    console.log(`${files.length} files found`)
    console.log("============ Converting files to tensors ============")

    let ys = []
    let xs = []

    files.forEach((file) => {
        const images = fs.readFileSync(file)
        const answer = encodeDirectory(file)
        const imageTensor = tf.node.decodeImage(images, 1)

        ys.push(answer)
        xs.push(imageTensor)
    })

    return [ys, xs]
}

// Because we're in ML environment, it's better to refer to a group as a number.
// In this case where we are matching image with a label, this is a valid way doing it.
function encodeDirectory(filePath) {
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

function shuffleTensors(array01, array02) {
    let counter = array01.length
    // If the assert is false the error log is written to the console
    console.assert(array01.length === array02.length)
    let temp01 = null,
        temp02 = null,
        index = 0

    while (counter > 0) {
        index = (Math.random() * counter) | 0
        counter--

        temp01 = array01[counter]
        temp02 = array02[counter]
        array01[counter] = array01[index]
        array02[counter] = array02[index]
        array01[counter] = temp01
        array02[counter] = temp02
    }
}

function normalizeImageData(xs, ys) {
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

    return [xNormalized, y]
}

export function saveProgression(value) {
    console.log(`
    ╔════════════╗
    ║   SAVING   ║
    ║   %${(value * 100).toFixed(2)}   ║
    ╚════════════╝
    `)
}

// For some reason this function is never executed if imported in index.js
export async function saveBestValidModel(model, path, best) {
    return {
        onEpochEnd: async (_epoch, logs) => {
            if (logs.val_acc > best) {
                saveProgression(logs.val_acc)
                model.save(path)
                best = logs.val_acc
            }
        }
    }
}
import * as tf from "@tensorflow/tfjs"
import { CLASSES } from "./labels.js"

async function performDetection() {
    // Majority of this function can be refactored to handle tensors in
    // a single patch and/or iterating through a list of elements and tensors.
    await tf.ready()
    const modelPath = "https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1"
    const model = await tf.loadGraphModel(modelPath, { fromTFHub: true })

    const image01 = document.getElementById("image-topk")
    const image02 = document.getElementById("image-advanced")
    const tensor01 = tf.browser.fromPixels(image01)
    const tensor02 = tf.browser.fromPixels(image02)

    // SSD Mobilenet single batch. These batches could also be put together.
    const singleBatch01 = tf.expandDims(tensor01, 0)
    const singleBatch02 = tf.expandDims(tensor02, 0)
    const imageResults01 = await model.executeAsync(singleBatch01)
    const imageResults02 = await model.executeAsync(singleBatch02)

    const canvasDetails01 = prepareCanvas("image-result-topk", image01)
    const canvasDetails02 = prepareCanvas("image-result-advanced", image02)
    // Get a clean tensor of indices
    const results01 = await cleanDetectionSetup(imageResults01, 5)
    showDetectionResultsOnCanvas(results01, canvasDetails01)

    const nmsResults = await cleanDetectionSetup(imageResults02, 10, true)

    tf.dispose([
        model,
        tensor01,
        tensor02,
        results01,
        nmsResults,
        singleBatch01,
        singleBatch02,
        imageResults01,
        imageResults02,
        imageResults02,
    ])

    console.log("Tensor memory status:", tf.memory().numTensors)
}

function prepareCanvas(imageId, source) {
    const canvas = document.getElementById(imageId)
    const context = canvas.getContext("2d")
    const imgWidth = source.width
    const imgHeight = source.height
    canvas.width = imgWidth
    canvas.height = imgHeight

    return {context, imgHeight, imgWidth}
}

async function cleanDetectionSetup(results, maxBoxes = 20, nms = false) {
    // Maxboxes, to limit the amount of detection boxes
    const prominentDetection = tf.topk(results[0])
    const topVals = prominentDetection.values.squeeze()
    const topDetections = tf.topk(topVals, maxBoxes)

    // Move results back to JavaScript in parallel
    const [maxDetections, maxIndices, maxValues, boxes] = await Promise.all([
        topDetections.indices.array(),
        prominentDetection.indices.data(),
        prominentDetection.values.data(),
        results[1].squeeze().array(),
    ])

    if (nms) {
        return await nmsDetection(results[1].squeeze(), topVals, maxBoxes, 0.5, 0.3)
    }

    return [maxDetections, maxIndices, maxValues, boxes]
}

async function nmsDetection(boxes, values, maxBoxes, iouThreshold, threshold) {
    const nmsDetection = await tf.image.nonMaxSuppressionWithScoreAsync(
        boxes, values, maxBoxes, iouThreshold, threshold, 1
    )
    const chosen = await nmsDetection.selectedIndices.data()

    return chosen
}

function showDetectionResultsOnCanvas(detectionResults, imageDetails, threshold = 0.3) {
    // Threshold, results below 30% should be ignored by default
    const [maxDetections, maxIndices, maxValues, boxes] = detectionResults
    const {context, imgHeight, imgWidth} = imageDetails

    maxDetections.forEach(detection => {
        context.strokeStyle = "#0F0"
        context.lineWidth = 1

        const detectedIndex = maxIndices[detection]
        const detectedClass = CLASSES[detectedIndex]
        const detectedScore = maxValues[detection]
        const dBox = boxes[detection]

        if (detectedScore > threshold) {
            // console.log(detectedClass, detectedScore)
            const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0
            const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0
            const height = (dBox[2] - dBox[0]) * imgHeight
            const width = (dBox[3] - dBox[1]) * imgWidth
            context.strokeRect(startX, startY, width, height)
        }
    })
}

function showNMSDetectionsResultsOnCanvas(chosen, imageDetails) {
    const {context, imgHeight, imgWidth} = imageDetails

    chosen.forEach(detection => {
        context.strokeStyle = "#0F0"
        context.lineWidth = 4
        const detectedIndex = maxIndices[detection]
        const detectedClass = CLASSES[detectedIndex]
        const detectedScore = scores[detection]
        const dBox = boxes[detection]

        // No negative values for start positions
        const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0
        const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0
        const height = (dBox[2] - dBox[0]) * imgHeight
        const width = (dBox[3] - dBox[1]) * imgWidth
        context.strokeRect(startX, startY, width, height)
    })
}

performDetection()

// tf.ready().then(() => {
//     const processing = document.getElementById("processing")
//     processing.innerHTML = "Processing . . ."

//     const modelPath = "https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1"

//     const imageElements = document.querySelectorAll(".image")
//     const images = []
//     imageElements.forEach(image => images.push(image))

//     tf.tidy(() => {
//         tf.loadGraphModel(modelPath, { fromTFHub: true })
//             .then(async model => {
//                 const singleImageTensor = await turnSourceIntoTensor("singleImage", images[0])

//                 // Create batch of data (with a single item)
//                 const batch = tf.expandDims(singleImageTensor, 0)
//                 const results = await model.executeAsync(batch)
//                 console.log("RESULT", results)
//                 // const detectionBoxes = objectDetection("topk", results)
//                 showResultOnCanvas("result", images[0], results)

//                 // Using video source
//                 // const streamTensor = getSourceData("videocam", )

//                 // const streamBatch = streamTensor.expandDims(0)
//                 // const streamResults = await model.executeAsync(streamBatch)

//                 // requestAnimationFrame(() => {
//                 //     const videoDetectionBoxes = objectDetection("nms", streamResults)
//                 //     showResultOnCanvas("result", images[0], videoDetectionBoxes)
//                 // })
//             })
//             .finally(() => {
//                 processing.innerHTML = ""
//             })
//     })
// })

// async function turnSourceIntoTensor(sourceType = "singleImage", source) {
//     if (sourceType === "singleImage") {
//         return tf.browser.fromPixels(source)
//     }

//     else if (sourceType === "webcam") {
//         // Use webcam as the source, such as webcam
//         try {
//             const videoStream = await receiveWebcamStream()
//             return tf.browser.fromPixels(videoStream)
//         } catch (err) {
//             console.warn("Error occured while loading video stream.\n", err)
//             return { error: err }
//         }
//     }
// }

// async function receiveWebcamStream(videoRef) {
//     if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
//         const webcamStream = await navigator.mediaDevices.getUserMedia({
//             audio: false,
//             video: {
//                 facingMode: "user",
//             },
//         })

//         if ("srcObject" in videoRef)
//             videoRef.srcObject = webcamStream
//         else
//             videoRef.src = window.URL.createObjectURL(webcamStream)

//         return new Promise((resolve, _) => {
//             videoRef.onloadedmetadata = () => {
//                 const canvas = document.getElementById("video-result")
//                 const context = canvas.getContext("2d")

//                 const imageWidth = image.width
//                 const imageHeight = image.height
//                 canvas.width = imageWidth
//                 canvas.height = imageHeight
//                 context.font = "16px sans-serif"
//                 context.textBaseline = "top"

//                 resolve([context, imageHeight, imageWidth])
//             }
//         })
//     }
//     else {
//         console.log("Video feed cannot be retreived")
//         alert("No webcam detected.")
//     }
// }

// async function objectDetection(mode = "basic", results) {
//     const detectionThreshold = 0.4
//     const iouThreshold = 0.5
//     const maxBoxes = 20

//     if (mode === "basic") {
//         // If you want to see how chaotic the detection can be,
//         // pass the variable below to setupCanvas as the third argument.
//         return await results[1].squeeze().array()
//     }

//     else if (mode === "topk") {
//         // Cleaning up the results
//         // 1) Find the highest ranked results/predictions
//         // Supress all other results that have lower predictions "score",
//         // which can be done easily utilizing topk function.
//         const acceptableDetectionBoxes = tf.topk(results[0])
//         const boxes = results[1].squeeze()
//         const values = acceptableDetectionBoxes.values.squeeze()

//         const [maxIndices, scores, detectionBoxes] = await Promise.all([
//             acceptableDetectionBoxes.indices.data(),
//             values.array(),
//             boxes.array(),
//         ])

//         return [maxIndices, scores, detectionBoxes]
//     }

//     else if (mode === "nms") {
//         // When objects overlap or are close enough, the previous example
//         // may not work as intended in all scenarios, which is why it"s
//         // recommended to use soft-NMS function in object detection.
//         const nmsDetection = await tf.image.nonMaxSuppressionWithScoreAsync(
//             boxes,
//             values,
//             maxBoxes,
//             iouThreshold,
//             detectionThreshold,
//             1 // Which NMS should be used. 0 normal, 1 soft.
//         )

//         const chosen = await nmsDetection.selectedIndices.data()
//     }
// }

// async function showResultOnCanvas(canvasId = "result", image, boxes) {
//     const canvas = document.getElementById(canvasId)
//     const context = canvas.getContext("2d")

//     // Make sure the canvas is clear from older detection boxes.
//     context.clearRect(0, 0, context.canvas.width, context.canvas.height)

//     const imageWidth = image.width
//     const imageHeight = image.height
//     canvas.width = imageWidth
//     canvas.height = imageHeight

//     boxes.forEach((detectedObject, i) => {
//         // Border
//         context.strokeStyle = "#F00"
//         context.globalCompositeOperation = "destination-over"
//         const startY = detectedObject[0] > 0 ? detectedObject[0] * imageHeight : 0
//         const startX = detectedObject[1] > 0 ? detectedObject[1] * imageWidth : 0
//         const height = (detectedObject[2] - detectedObject[0]) * imageHeight
//         const width = (detectedObject[3] - detectedObject[1]) * imageWidth
//         context.strokeRect(startX, startY, width, height)

//         // Label
//         context.globalCompositeOperation = "source-over"
//         context.fillStyle = "#E00"
//         context.lineWidth = 2
//         context.font = "16px sans-serif"
//         context.textBaseline = "top"

//         const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`
//         const textWidth = context.measureText(label).width
//         const textHeight = 16
//         const textPad = 4

//         context.fillRect(
//             startX,
//             startY,
//             textWidth + textPad,
//             textHeight + textPad
//         )

//         context.fillStyle = "#000000"
//         context.fillText(label, startX, startY)
//     })
// }
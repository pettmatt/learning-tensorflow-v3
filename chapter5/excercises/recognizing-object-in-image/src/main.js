import * as tf from "@tensorflow/tfjs"

tf.ready().then(() => {
    const images = document.querySelectorAll("img")
    const predicts = document.querySelectorAll(".prediction")
    const croppedCanvases = document.querySelectorAll(".cropped")
    const processing = document.getElementById("processing")
    processing.innerHTML = "Processing . . ."

    const modelPath = "/model/tfjs_quant_uint8/model.json"

    tf.tidy(() => {
        const result = tf.loadLayersModel(modelPath)
            .then(model => {
                const tensors = []
                images.forEach(image => { 
                    const tensor = tf.browser.fromPixels(image)
                    tensors.push(tensor)
                })

                // Reshaping to expected form
                const reshaped = tensors.map(tensor => 
                    tf.image
                        .resizeNearestNeighbor(tensor, [256, 256], true)
                        .div(255)
                        // Notice how reshape is working here without a problem, compared to thub-as-part... excercise
                        .reshape([1, 256, 256, 3])
                )

                const predictions = reshaped.map((tensor, i) => {
                    const result = model.predict(tensor)
                    result.print()
                    drawBoxOnCanvas(images[i], predicts[i], result)
                    return result
                })

                // Excercise: Extract the pet's face from image. Create separate model that expects image of size 96x96
                const cropped = calculateImageCropping(tensors[0], predictions[0])

                const croppedFace = tf.image
                    .resizeBilinear(cropped, [96, 96], true)
                    .reshape([1, 96, 96, 3])

                tf.browser.toPixels(croppedFace, croppedCanvases[0])

                processing.innerHTML = ""
            })
            .catch(err => {
                console.error("Error occured with model: " + modelPath + "'.\n", err)
                processing.innerHTML = err
            })
        return result
    })

})

function drawBoxOnCanvas(image, prediction, modelPrediction) {
    const imgWidth = image.width
    const imgHeight = image.height
    prediction.width = imgWidth
    prediction.height = imgHeight

    const box = modelPrediction.dataSync()
    const startX = box[0] * imgWidth
    const startY = box[1] * imgHeight
    const width = (box[2] - box[0]) * imgWidth
    const height = (box[3] - box[1]) * imgHeight

    const ctx = prediction.getContext("2d")
    ctx.strokeStyle="#0FE"
    ctx.lineWidth = 4
    ctx.strokeRect(startX, startY, width, height)
}

// Book example how to solve the cropping excercise
function calculateImageCropping(tensor, prediction) {
    const tensorHeight = tensor.shape[0]
    const tensorWidth = tensor.shape[1]

    const box = prediction.dataSync()
    const tensorStartX = box[0] * tensorWidth
    const tensorStartY = box[1] * tensorHeight
    const cropLength = parseInt((box[2] - box[0]) * tensorWidth, 0)
    const cropHeight = parseInt((box[3] - box[1]) * tensorHeight, 0)

    const startPosition = [tensorStartY, tensorStartX, 0]
    const cropSize = [cropHeight, cropLength, 3]

    const cropped = tf.slice(tensor, startPosition, cropSize)

    return cropped
}
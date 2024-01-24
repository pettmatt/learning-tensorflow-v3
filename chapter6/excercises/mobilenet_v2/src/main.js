import * as tf from "@tensorflow/tfjs"
import { INCEPTION_CLASSES } from "./labels"

tf.ready().then(() => {
    const predicts = document.querySelectorAll(".prediction")
    const processing = document.getElementById("processing")
    processing.innerHTML = "Processing . . ."

    const modelPath = "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TfJs/variations/035-128-classification/versions/3"

    tf.tidy(() => {
        tf.loadGraphModel(modelPath, { fromTFHub: true })
            .then(model => {
                const imageElements = document.querySelectorAll(".image")
                const images = []
                imageElements.forEach(image => images.push(image))

                processing.innerHTML = "Creating tensors . . ."
                const tensors = images.map(image =>
                    tf.browser.fromPixels(image)
                )

                const resizedImageTensors = tensors.map(t =>
                    tf.image
                        .resizeBilinear(t, [128, 128], true)
                        .div(128)
                        .reshape(1, 128, 128, 3)
                        // .expandDims(0)
                )

                // Creating a batch
                const imageBatch = tf.expandDims(resizedImageTensors[0], 0)

                model.executeAsync(imageBatch).then((result) => {
                    console.log("first", result[0].shape)
                    result[0].print()
                    console.log("second", result[1].shape)
                    result[1].print()
                })

                // Resizing to match what model expects, which is 299x299
                // processing.innerHTML = "Reshaping tensor(s)"
                // const resizedImageTensors = tensors.map(t =>
                //     tf.image
                //         .resizeBilinear(t, [299, 299], true)
                //         .div(255)
                //         // For some reason the reshape is unable to convert the image to expected dimensions at this moment.
                //         // .reshape(1, 299, 299, 3)
                //         // Which is why reshape is replaced with expandDims.
                //         .expandDims(0)
                // )

                // console.log("resizedImageTensors", resizedImageTensors)

                // const results = resizedImageTensors.map(resizedT => {
                //     const r = model.predict(resizedT)
                //     r.print()
                //     return r
                // })


                // const indices = results.map(result => {
                //     const { values, indices } = tf.topk(result, 3)
                //     indices.print()
                //     return indices
                // })

                // const predictionList = indices.map(indice => indice.dataSync())
                // const listedPredictions = predictionList.map((predictions, i) => {
                //     const imagePredicts = predictions.map((p, ii) => {
                //         // Because the model doesn't include labels, the program needs to match the prediction with the labels.
                //         const label = INCEPTION_CLASSES[p]
                //         console.log(`Prediction (${ii + 1}, image ${i + 1}):`, label)

                //         if (predicts[i].innerHTML === "")
                //             predicts[i].innerHTML = label
                //         else predicts[i].innerHTML += ", " + label

                //         return label
                //     })

                //     return imagePredicts
                // })
                // console.log(listedPredictions)

                processing.innerHTML = ""
            })
            .catch(err => {
                console.error("Error occured with model: " + modelPath + "'.\n", err)
                processing.innerHTML = err
            })
    })
})

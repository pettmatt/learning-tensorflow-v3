import * as tf from "@tensorflow/tfjs"

tf.ready().then(() => {
    const modelPath = "model/ttt_model.json"

    tf.tidy(() => {
        tf.loadLayersModel(modelPath)
            .then(model => {
                // Board states. Could be great practice to generate states from images
                const emptyBoard = tf.zeros([9])
                const blockOpponent = tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1])
                const winningBoard = tf.tensor([-1, 0, -1, 0, 1, 1, 1, 0, -1])

                // Stack into a shape
                const matches = tf.stack([emptyBoard, blockOpponent, winningBoard])
                const result = model.predict(matches)
                // Logging (reshaping for increasing the readability)
                result.reshape([3, 3, 3]).print()
            })
            .catch(err => {
                console.error("Loading a model failed. Model path '" + modelPath + "'.\n", err)
            })
    })
})

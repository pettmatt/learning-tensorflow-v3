window.addEventListener("load", function () {
    const canvas = document.querySelector("#sorted-canvas")

    const randomStaticValues = tf.randomUniform([400, 400], 0, 255, "int32")

    // Manipulate the pixels
    const sorted = tf.topk(randomStaticValues, 400).values
    const reshaped = sorted.reshape([400, 400, 1])

    tf.browser.toPixels(reshaped, canvas)
        .then(() => {
            sorted.dispose()
            reshaped.dispose()
            randomStaticValues.dispose()
        })

    console.log("Should be clean environment", tf.memory().numTensors)
})
window.addEventListener("load", function () {
    flipImage()
    // flipImageBatch()

    resizeImage()

    cropImage()
})

function flipImage() {
    const flippedImage = document.querySelector("#flipped-image")
    const flippedCanvas = document.querySelector("#flipped-canvas")
    const tensor = tf.browser.fromPixels(flippedImage)

    // Flip an single image
    const flipped = tf.reverse(tensor, 1)

    // Produce a result, in this case the data is thrown onto a canvas
    tf.browser.toPixels(flipped, flippedCanvas)
        .then(() => {
            tensor.dispose()
            flipped.dispose()
        })
}

function flipImageBatch() {
    const flippedImage = document.querySelector("#flipped-image")
    const flippedCanvas = document.querySelector("#flipped-canvas")
    const tensor = tf.browser.fromPixels(flippedImage)

    // Flip a batch of images, this example contains a batch of one image
    const batch = tf.expandDims(
        tensor.asType("float32")
    ) // Creates a list of tensors (4D tensor)
    // Squeeze removes one layer from the tensor, in this case it creates 3D tensor
    const flipped = tf
        .squeeze(tf.image.flipLeftRight(batch))
        .asType("int32")

    // Produce a result, in this case the data is thown onto a canvas
    tf.browser.toPixels(flipped, flippedCanvas)
        .then(() => {
            batch.dispose()
            tensor.dispose()
            flipped.dispose()
        })
}

function resizeImage() {
    const resizeTo = [768, 560] // 4x bigger than original
    const resizeImage = document.querySelector("#resized-image")
    const resizeCanvas01 = document.querySelector("#resized-canvas-01")
    const resizeCanvas02 = document.querySelector("#resized-canvas-02")

    const tensor = tf.browser.fromPixels(resizeImage)

    // Pixelated
    // First example with nearest neighbor
    const resized01 = tf.image.resizeNearestNeighbor(tensor, resizeTo, true)
    const resized01Int = resized01.asType("int32")

    tf.browser.toPixels(resized01Int, resizeCanvas01)
        .then(() => {
            resized01.dispose()
            resized01Int.dispose()
        })

    // Blurred
    // Now let's use resize bilinear. These algorithms have slightly different result
    const resized02 = tf.image.resizeBilinear(tensor, resizeTo, true)
    const resized02Int = resized02.asType("int32")

    tf.browser.toPixels(resized02Int, resizeCanvas02)
        .then(() => {
            resized02.dispose()
            resized02Int.dispose()
        })

    tensor.dispose()
}

// Optional exercise: Resize checker 4x3 example
// function resizeChecker() { }

// If image fails to load, your screen is too small
function cropImage() {
    const startFrom = [150, 230, 0] // 0 pixels down, 40 pixels over, red channel
    const cropSize = [150, 150, 3] // Y, X axis in pixels, and finally all color channels included
    const cropImage = document.querySelector("#cropped-image")
    const cropCanvas = document.querySelector("#cropped-canvas")

    const tensor = tf.browser.fromPixels(cropImage)
    const cropped = tf.slice(tensor, startFrom, cropSize)

    tf.browser.toPixels(cropped, cropCanvas).then(() => {
        cropped.dispose()
    })

    tensor.dispose()
}
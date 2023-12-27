window.addEventListener("load", function () {
    // flipImage()
    flipImageBatch()
})

function flipImage() {
    console.log("Flipping image")
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
    console.log("Flipping batch of image(s)")
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
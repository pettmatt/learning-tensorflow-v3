import * as tf from "@tensorflow/tfjs-node"

export default function getModel() {
    const model = tf.sequential()

    model.add(
        tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            strides: 1,
            padding: "same",
            activation: "relu",
            kernelInitializer: "heNormal",
            inputShape: [28, 28, 1]
        })
    )

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))

    model.add(
        tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            strides: 1,
            padding: "same",
            activation: "relu"
        })
    )

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))

    model.add(
        tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            strides: 1,
            padding: "same",
            activation: "relu"
        })
    )

    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))

    // Flatten for connecting to deep layers
    model.add(tf.layers.flatten())

    // Hidden layer
    model.add(
        tf.layers.dense({
            units: 128,
            activation: "tanh"
        })
    )

    // Output
    model.add(
        tf.layers.dense({
            units: 10,
            activation: "softmax"
        })
    )

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    })

    console.log("================== Model creation completed ==================")
    model.summary()

    return model
}
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const tf = __importStar(require("@tensorflow/tfjs-node"));
function getModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        strides: 1,
        padding: "same",
        activation: "relu",
        kernelInitializer: "heNormal",
        inputShape: [28, 28, 1]
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        strides: 1,
        padding: "same",
        activation: "relu"
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        strides: 1,
        padding: "same",
        activation: "relu"
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.flatten());
    // Hidden layer
    model.add(tf.layers.dense({
        units: 128,
        activation: "tanh"
    }));
    // Output
    model.add(tf.layers.dense({
        units: 10,
        activation: "softmax"
    }));
    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });
    console.log("================== Model creation completed ==================");
    model.summary();
    return model;
}
exports.default = getModel;

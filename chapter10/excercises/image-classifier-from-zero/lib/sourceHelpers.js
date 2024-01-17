"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.folderToTensors = void 0;
const tfjs_node_1 = __importDefault(require("@tensorflow/tfjs-node"));
const fs_1 = __importDefault(require("fs"));
function folderToTensors() {
    return __awaiter(this, void 0, void 0, function* () {
    });
}
exports.folderToTensors = folderToTensors;
function filesToTensors(files) {
    let ys = [];
    let xs = [];
    files.forEach((file) => {
        const images = fs_1.default.readFileSync(file);
        const answer = encodeDirectory(file);
        const imageTensor = tfjs_node_1.default.node.decodeImage(images, 1);
        ys.push(answer);
        xs.push(imageTensor);
    });
    return normalizeImageData(xs, ys);
}
// Because we're in ML environment, it's better to refer to a group as a number.
// In this case where we are matching image with a label, this is a valid way doing it.
function encodeDirectory(filePath) {
    if (filePath.includes("owl"))
        return 0;
    if (filePath.includes("bird"))
        return 1;
    if (filePath.includes("lion"))
        return 2;
    if (filePath.includes("snail"))
        return 3;
    if (filePath.includes("snake"))
        return 4;
    if (filePath.includes("skull"))
        return 5;
    if (filePath.includes("tiger"))
        return 6;
    if (filePath.includes("parrot"))
        return 7;
    if (filePath.includes("raccoon"))
        return 8;
    if (filePath.includes("squirrel"))
        return 9;
    console.error("Unrecognized folder");
    process.exit(1);
}
// interface TensorDatasetObject {
//     xs: Tensor4D,
//     ys: Tensor4D,
// }
function normalizeImageData(xs, ys) {
    console.log("============ Normalizing image data ============");
    console.log("Stacking");
    const x = tfjs_node_1.default.stack(xs);
    const y = tfjs_node_1.default.oneHot(ys, 10);
    console.log("All images have been converted to tensors:");
    console.log("x", x.shape);
    console.log("y", y.shape);
    console.log("Finally, normalizing x values");
    const xNormalized = x.div(255);
    tfjs_node_1.default.dispose([x]);
    return { xs: xNormalized, ys: y };
}

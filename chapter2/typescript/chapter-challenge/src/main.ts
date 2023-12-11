import "@tensorflow/tfjs"
import * as mobilenet from "@tensorflow-models/mobilenet"

const input = document.querySelector("input[type='file']") as HTMLInputElement
const form = document.querySelector("form") as HTMLFormElement
const element = document.querySelector("#results-container") as HTMLDivElement

input.addEventListener("input", (event) => {
    const input = event.target as HTMLInputElement
    let imageCount = 0

    for (let i = 0; i < (input.files?.length || 0); i++) {
        imageCount++
        const file = input.files?.[i]

        const reader = new FileReader()
        reader.addEventListener("load", () => {
            if (typeof reader.result === "string") {
                const image = new Image()
                image.width = 230
                image.title = file?.name || "undefined_title"
                image.src = reader.result
                addImage(image)
            }
        }, false)

        if (file) reader.readAsDataURL(file)
    }
})

form.addEventListener("submit", (event) => {
    event.preventDefault()
    const images = document.querySelectorAll("img")

    images.forEach((image, i) => {
        checkIfTruck(image, i)
    })
})

function addImage(image: HTMLImageElement) {
    const imageContainer = document.querySelector("#image-container")
    if (imageContainer) imageContainer.appendChild(image)
}

function checkIfTruck(img: HTMLImageElement, index) {
    let result: any

    mobilenet.load().then(model => {
        if (img && img instanceof HTMLImageElement) {
            model.classify(img).then(predictions => {
                // console.log('Predictions: ', predictions)
                let foundATruck

                predictions.forEach(p => {
                    foundATruck = foundATruck || p.className.includes("truck")
                })

                return predictions
            })
            .then(isTruck => {
                if (element) {
                    element.innerHTML += `<h3>Image ${index + 1}</h3>`

                    isTruck.forEach((prob) => {
                        element.innerHTML += `
                            <li>
                                ${ Math.floor(prob.probability * 100) }% ${ prob.className }
                            </li>`
                    })
                }
            })
        }
    })

    return result
}
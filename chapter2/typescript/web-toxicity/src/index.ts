import "@tensorflow/tfjs"
import * as toxicity from "@tensorflow-models/toxicity"

// minimum positive prediction confidence
// If this isn't passed, the default is 0.85
const threshold: number = 0.5;

const answerContainer = document.querySelector("#answer-container")
const form = document.getElementById("comment-form")

form?.addEventListener("submit", (event) => {
  event.preventDefault()

  if (answerContainer) answerContainer.innerHTML = "Processing . . ."

  if (event.target !== null) {
    const inputValue = event.target[0].value
    processCommentWithML(inputValue)
  }
  else console.log("Element structure has changed")
})

function processCommentWithML(sentence: String) {
  toxicity.load(threshold, []).then((model) => {
    const sentences = [sentence.toString()]
  
    model.classify(sentences).then((predictions) => {
      // semi-pretty-print results
      const JSONprediction = JSON.stringify(predictions, null, 2)
      console.log(JSONprediction)

      const list = predictions.map((item) => 
        "<li><b>" + item.label + "</b> " +
        JSON.stringify(item.results[0]) +
        "</li>"
      )

      if (answerContainer) answerContainer.innerHTML = list.join("")
    })
  })
}
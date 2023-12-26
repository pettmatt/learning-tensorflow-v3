import * as tf from "@tensorflow/tfjs"
// import * as tf from "@tensorflow/tfjs-node-gpu"

const users = ["User 1", "User 2", "User 3"]
const bands = ["Linkin Park", "Nirvana", "Three Nights"]
const genres = ["Rock", "Pop", "Industrial"]
// User preference based on their input on bands and songs that were prompted
const userVotes = tf.tensor([
    [10, 9, 1],
    [0, 1, 3],
    [7, 9, 8],
])
// Music styles
const bandGenres = tf.tensor([
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
])

// userVotes: Array<number[]>, bands: string[], genres: string[]
export default function main() {
    const userGenres = tf.matMul(userVotes, bandGenres)
    userGenres.print()

    // Get the top result
    const topUserGenres: any = tf.topk(userGenres, genres.length)

    // transform the data into something that is readable in JavaScript
    const topGenres = topUserGenres.indices.arraySync()
    users.map((user, index) => {
        const rankedCategories = topGenres[index].map((v: any) => genres[v])
        // New recommendations based on best rated genres
        console.log(user, rankedCategories)
    })
}
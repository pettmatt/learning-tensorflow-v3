import express from "express"

const app = express()

app.get("/api/spotify-accounts", (req, res) => {

})

app.get("/api/spotify", (req, res) => {
    const fetchRes = fetch("https://accounts.spotify.com")
})

app.listen(5000, () => 
    console.log(`Waiting for requests at http://localhost:5000`)
)
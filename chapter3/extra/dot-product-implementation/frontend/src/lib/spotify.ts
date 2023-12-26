const applicationScope = ["user-read-playback-position", "user-top-read", "user-read-recently-played"]
const clientId = import.meta.env.VITE_SPOTIFY_CLIENT_ID
const redirectUri = import.meta.env.VITE_SPOTIFY_REDIRECT_URI
const clientSecret = import.meta.env.VITE_SPOTIFY_CLIENT_SECRET

const tokenExpirationTime = 3600 * 1000

export async function login() {
    const res = await fetch("https://accounts.spotify.com" +
        `?response_type=code` +
        `&client_id=${clientId}` +
        `&redirect_uri=${redirectUri}` +
        `&scope=${applicationScope.join(" ")}`,
        {
            mode: "no-cors"
        }
    )

    if (!res.ok) console.log("User authorization was not OK", res)

    const data = await res.json()
    console.log("LOGIN DATA", data)

    // Exchange the auth code for access token
    // const tokenRes = await fetch("https://accounts.spotify.com/api/token", {
    //     method: "POST",
    //     headers: {
    //         "Content-Type": "application/x-www-form-urlencoded",
    //         "Authorization": "Basic "
    //     },
    //     body: JSON.stringify({
    //         // code: data.code,
    //         client_id: clientId,
    //         grant_type: "authorization_code",
    //         redirect_uri: redirectUri,
    //     })
    // })

    // if (!tokenRes.ok) console.log("Token request was not OK")

    // console.log("Token", await tokenRes.json())
}

// export function async getUserAccessToken() {
//     const authRes = await fetch("https://api.spotify.com/v1")
//     // const authorized = 
// }

export function logOut() {

}

export async function getTopSongs(token: string) {
    const config = {
        Authorization: `Bearer`
    }

    // const authRes = await fetch("https://api.spotify.com/v1", config)
}
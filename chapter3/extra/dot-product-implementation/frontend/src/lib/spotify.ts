const serverUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:3010"

export async function makeApiRequest(path: string) {
    const token = getLocalToken()
    const res = await fetch(`${serverUrl}${path}`, {
        headers: { Authorization: token }
    })

    const body = await res.json()

    return body
}

function getLocalToken(): string {
    const tokenLS = localStorage.getItem("token") || ""
    const tokenObject = JSON.parse(tokenLS)
    return `${tokenObject.token_type} ${tokenObject.access_token}` || ""
}

export async function getAccessToken(code: string | null, state: string | null) {
    const response = await fetch(`${serverUrl}/spotify/token`, {
        method: "POST",
        body: JSON.stringify({
            code,
            state,
        })
    })

    const body = await response.json()

    if (!response.ok) {
        return {
            error: body
        }
    }

    return { token: body.data }
}

export function validateSessionToken(token: object) {
    localStorage.setItem("token", JSON.stringify(token))

    const currentTime = new Date().getTime()
    localStorage.setItem("token-creation-time", JSON.stringify(currentTime))

    const validated = checkSession()

    return validated
}

export function checkSession() {
    const currentTime = new Date().getTime()
    const currentTimeInSeconds = currentTime / 1000
    const tokenLS = localStorage.getItem("token")
    const token = tokenLS ? JSON.parse(tokenLS) : ""

    if (!token) return false

    const tokenCreationTimeLS = localStorage.getItem("token-creation-time")

    if (!tokenCreationTimeLS) return false

    const tokenCreationTime = Number(tokenCreationTimeLS) / 1000
    const expired = (tokenCreationTime + token.expires_in) < currentTimeInSeconds

    if (expired) return false

    return true
}
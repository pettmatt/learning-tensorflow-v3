export function generateRandomString(stringLength: number): String {
    if (stringLength % 2 == 1)
        throw new Deno.errors.InvalidData("Only even sizes are supported")

    const buf = new Uint8Array(stringLength / 2)
    crypto.getRandomValues(buf)
    let randomString = ""

    for (let i = 0; i < buf.length; ++i) {
        randomString += ("0" + buf[i].toString(16)).slice(-2)
    }

    return randomString
}

export function generateCodeVerifier(length: number) {
    let text = ""
    let possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    for (let i = 0; i < length; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length))
    }
    return text
}

export async function generateCodeChallenge(codeVerifier: string) {
    const data = new TextEncoder().encode(codeVerifier)
    const digest = await window.crypto.subtle.digest('SHA-256', data)
    return btoa(String.fromCharCode.apply(null, [...new Uint8Array(digest)]))
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '')
}
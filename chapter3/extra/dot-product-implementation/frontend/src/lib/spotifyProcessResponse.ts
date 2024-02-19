export function processSpotifyResponse(response: any) {
    const responseObject = (response.error)
        ? handleErrorResponse(response)
        : handlePositiveResponse(response)

    return responseObject
}

function handlePositiveResponse(response: any) {
    return response
}

function handleErrorResponse(response: any) {
    if (response.error.status === 401) {
        return { error: { message: "Bad or expired token.", s_message: response.error.message } }
    } else if (response.error.status === 403) {
        return { error: { message: "Bad OAuth request. API error.", s_message: response.error.message } }
    } else if (response.error.status === 429) {
        return { error: { message: "Application has  exceeded the rate limit. Try again later.", s_message: response.error.message } }
    } else {
        return { error: response }
    }
}
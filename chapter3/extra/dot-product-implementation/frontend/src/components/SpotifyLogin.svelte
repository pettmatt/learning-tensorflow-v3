<script lang="ts">
    import { validateSessionToken, checkSession, getAccessToken } from "../lib/spotify"
    import SpotifyProfile from "./SpotifyProfile.svelte"

    let authenticationError: null | any = null
    let authentication: boolean = (localStorage.getItem("token") ? true : false)
    let pageUri: string = window.location.href

    const url = new URL(pageUri)
    const error = url.searchParams.get("error")
    const codeParam = url.searchParams.get("code") || null
    const stateParam = url.searchParams.get("state") || null

    async function validateSession() {
        const sessionIsValid = checkSession()

        if (sessionIsValid) return // Nothing to do here. Session should be usable.

        if (error)
            authenticationError = { error }

        else if (pageUri?.includes("callback")) {
            getAccessToken(codeParam, stateParam)
                .then(res => {
                    // console.log("userauth res", res)
                    const token = res.token
                    authentication = validateSessionToken(token)
                })
                .catch(err => {
                    // "Session expired, please login again."
                    // "Authentication failed, please try again."
                    console.log("ERROR", err)
                    logout()
                })
        }
        else console.log("ExchangeAuthCode else clause... If you see this, something probably broke.")
    }

    function logout() {
        authentication = false
        localStorage.removeItem("token")
        localStorage.removeItem("token-creation-time")

        if (pageUri.includes("callback"))
            history.pushState(null, "", url.origin)
    }

    validateSession()
</script>

{#if authentication}
    <div class="profile-wrapper">
        <SpotifyProfile logout={ logout } />
    </div>
{:else if authenticationError}
    <div class="error-wrapper">
        <p>Login error: <span>{ authenticationError }</span></p>
        <p>Details: <span>{ authenticationError }</span></p>
    </div>
{:else}
    <div class="login-wrapper">
        <a class="button-like-link" href="http://localhost:3010/spotify/login">Login to Spotify</a>
        <small>Application requires login to Spotify</small>
    </div>
{/if}

<style>
    .login-wrapper {
        display: flex;
        flex-direction: column;
        gap: 0.5em;
    }
    .login-wrapper a {
        width: 8em;
        text-align: center;
        margin: 0 auto;
    }
    .login-wrapper small {
        text-align: center;
    }
</style>
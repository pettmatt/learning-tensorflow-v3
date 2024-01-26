<script lang="ts">
    import { makeApiRequest } from "../lib/spotify"
    import { processSpotifyResponse } from "../lib/spotifyProcessResponse"

    export let logout: any

    interface SpotifyRequestError {
        error: {
            message: string
            status: number
        }
    }

    let profile: SpotifyRequestError | any = null
    let showProfile = false

    async function getProfile() {
        const profileLS = localStorage.getItem("spotify-profile")
        if (profileLS) {
            profile = JSON.parse(profileLS)
        } else {
            const profileRes = await makeApiRequest("/spotify/u/profile")
            profile = processSpotifyResponse(profileRes)
            localStorage.setItem("spotify-profile", JSON.stringify(profile))
            console.log(profile)
        }
    }

    getProfile()
</script>

<nav class="profile-wrapper">
    {#if profile?.error}
        {profile.error.message}
    {:else}
        Logged in
        <button class="mini-button" on:click={() => { showProfile = !showProfile }}>
            { showProfile ? ">" : "v" }
        </button>
        {#if !showProfile}
            <a class="profile" href={profile?.external_urls.spotify}>
                <img class="profile-image" loading="lazy" src={profile?.images[0].url} alt="Profile" />
                <span>{profile?.display_name}</span>
            </a>
        {/if}

        <button class="link-like-button" on:click={ logout }>Logout</button>
    {/if}
</nav>

<style>
    .profile {
        display: flex;
        margin: 0.5em 1em;
        gap: 0.5em;
        padding: 0.25em;
        border: 2px solid none;
        border-radius: 2em;
        max-width: 12em;
        transform: ease 0.5s border;
    }
    .profile:hover {
        border: 2px solid var(--spotify-green);
        color: var(--spotify-green);
    }
    .profile-image {
        border-radius: 2em;
        width: 2rem;
    }
</style>
<script lang="ts">
    import { login } from "../lib/spotify"
    let logged: null | any = null
    const auth = () => {
        login() 
    }

    console.log("AUTH", logged)
</script>

{#if logged !== null}
    You're connected to Spotify!
{:else}
    <small>Requires logging in to Spotify</small>
    <button on:click={auth}>Login to Spotify</button>
{/if}
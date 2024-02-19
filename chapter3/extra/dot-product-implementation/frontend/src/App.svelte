<script lang="ts">
    import JsxGraph from "./components/JsxGraph.svelte"
    import tfMain from "./lib/tensorflow-base"
    import SpotifyLogin from "./components/SpotifyLogin.svelte"
    import Questionere from "./components/Questionere.svelte"
    import { checkSession } from "./lib/spotify"

    const showGraph = false
    let loggedIn = checkSession()
</script>

<main>
    <h1>Music recommender</h1>

    <SpotifyLogin />

    {#if loggedIn }

        <Questionere />

    {:else if showGraph}

        <JsxGraph />

    {/if}
</main>

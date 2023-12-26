<script lang="ts">
    import JsxGraph from "./components/JsxGraph.svelte"
    import tfMain from "./lib/tensorflow-base"
    import SpotifyLogin from "./components/SpotifyLogin.svelte";
    // tfMain()

    const showGraph = false
</script>

<main>
    <h1>Music recommender</h1>

    <SpotifyLogin />

    {#if !showGraph }

        <h2>Ye</h2>

    {:else if showGraph}

        <JsxGraph />

    {/if}
</main>

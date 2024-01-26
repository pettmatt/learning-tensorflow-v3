<script lang="ts">
    export let progress = {
        current: 0,
        ends: 3,
    }
</script>

<div class="progress-wrapper">
    <div>
        {#if progress.current === progress.ends}
            Time to generate your new playlist!
        {:else}
            { progress.current }
        {/if}
    </div>
</div>

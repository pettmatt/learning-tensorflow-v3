<script lang="ts">
    import { onDestroy, onMount } from "svelte"
    import { writable } from "svelte/store"
    import * as JXG from "jsxgraph"

    const boardStore = writable(null)
    let dotproduct: string = "66.00"

    onMount(() => {
        const boardInstance = JXG.JSXGraph.initBoard("jxg-box", {
            boundingbox: [-15, 15, 15, -15],
            axis: true,
        })

        boardStore.set(boardInstance)

        const subscribe = boardStore.subscribe(b => {
            if (b !== null) {
                const p1 = b.create("point", [-6, 8], {
                    name: "A",
                    size: 0,
                })
                const p2 = b.create("point", [0, 0], {
                    name: "",
                    size: 4,
                    fixed: true,
                })
                const p3 = b.create("point", [5, 12], {
                    name: "B",
                    size: 0,
                })

                const li1 = b.create("line", [p1, p2], {
                    straightFirst: false,
                    straightLast: false,
                    strokeWidth: 2,
                    dash: 2,
                    firstArrow: true,
                })

                const li2 = b.create("line", [p3, p2], {
                    straightFirst: false,
                    straightLast: false,
                    strokeWidth: 2,
                    dash: 2,
                    firstArrow: true,
                    strokeColor: "green",
                })

                b.on("update", function () {
                    dotproduct = (p1.X() * p3.X() + p1.Y() * p3.Y()).toFixed(2)
                })
            }
        })

        subscribe()
    })
    
    onDestroy(() => {
        const unsubscribe = boardStore.subscribe(boardInstance => {
            if (boardInstance) {
                boardInstance.removeEventHandlers()
                boardInstance.containerObj.innerHTML = ""
            }
        })

        unsubscribe()
    })
</script>

<div class="graph-wrapper">
    <h2>Test</h2>

    {#if dotproduct !== null}
        <h3>Dot product = {dotproduct}</h3>
    {/if}

    <div id="jxg-box" class="jxg-box">
    </div>
</div>

<style>
    #jxg-box {
        width: 100%;
        height: 500px;
    }

    .graph-wrapper {
        text-align: left;
    }
</style>
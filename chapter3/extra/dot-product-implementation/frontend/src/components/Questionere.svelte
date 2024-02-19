<!-- Question what the user thinks of these artists, songs and genres -->
<script lang="ts">
    import { type Artist, type Question, type Track } from "../lib/spotifyInterface"
    import QuestionereProgressBar from "./QuestionereProgressBar.svelte"
    import { makeApiRequest } from "../lib/spotify"

    const profile = {
        explicitContent: null,
    }
    let currentQuestion = 0
    let questions: Question[] = [
        {
            question: "Select genres",
            uri: "/spotify/genres",
            options: [],
            getOptions: async (): Promise<string>  => {
                const genresObject = await makeApiRequest(questions[currentQuestion].uri)
                const genres = genresObject.genres.map((g: string) =>
                    capitalizeReplaceDashesInString(g)
                )
                return genres.map((g: string) => 
                    g +
                    "<input type='checkbox' hidden=true name='" + g.toLocaleLowerCase() +"' value=false />"
                )
            },
            answers: [],
        },
        {
            question: "Do you want to listen artists that you follow?",
            subText: "Select all you wish to include.",
            uri: "/spotify/u/artists",
            options: [],
            getOptions: async (): Promise<string> => {
                const artistObject = await makeApiRequest(questions[currentQuestion].uri)
                const artists = artistObject.items
                return artists.map((a: Artist) => 
                    "<img src='" + a.images[2].url + "' alt='" + a.type + " profile image' />" +
                    "<h3>" + a.name + "</h3>" +
                    "<div class='genre-wrapper'>" +
                        "<ul>" +
                            a.genres.map((g: string) => (
                                "<li>" + capitalizeReplaceDashesInString(g) + "</li>"
                            )) +
                        "</ul>" +
                    "</div>"
                )
            },
            answers: [],
        },
        {
            question: "Do you still like these tracks?",
            subText: "Select all songs you like.",
            options: [],
            getOptions: async (): Promise<string> => {
                const trackObject = await makeApiRequest(questions[currentQuestion].uri)
                const tracks = trackObject.items
                return tracks.map((t: Track) =>
                    "<img src='" + t.album.images[1].url + "' alt='" + t.type + " image' />" +
                    "<h3>" + t.name + "</h3>" +
                    "<p>" + t.album.name + "</p>" +
                    "<div class='genre-wrapper'>" +
                        "<ul>" +
                            t.artists.map((a: Artist) => 
                                "<li>" +
                                    a.name +
                                "</li>"
                            ) +
                        "</ul>" +
                    "</div>"
                )
            },
            uri: "/spotify/u/tracks",
            answers: [],
            followUp: [],
        },
    ]

    function capitalizeReplaceDashesInString(s: string): string {
        return s.charAt(0).toUpperCase() + s.slice(1).replaceAll("-", " ")
    }
    
    async function progress() {
        // const followUp = questions[currentQuestion]?.followUp
        // const previousAnswer = questions[currentQuestion]?.answers

        currentQuestion < questions.length
            ? currentQuestion++
            : null
    }

    function toggleAnswer(innerHTML: string) {
        const answers = questions[currentQuestion].answers
        const startOfInput = innerHTML.indexOf("<")
        const name = innerHTML.slice(0, startOfInput)

        if (answers.includes(name))
            questions[currentQuestion].answers = answers.filter(a => a !== name)
        else
            questions[currentQuestion].answers.push(name)

        answers.map(a => {
            const input = document.querySelector(`input[name="${name}"]`)
            input!.value === "false"
                ? input!.value = "true"
                : input!.value = "false"
        })
    }

    async function load(){
        const profileLS = localStorage.getItem("spotify-profile") || null
        const profile = (profileLS) && JSON.parse(profileLS)

        if (profile) {
            profile.explicitContent = profile.explicit_content.filter_enabled
        }
    }

    load()
</script>

<div class="questionere-wrapper">
    <div class="header-wrapper">
        <header>
            <QuestionereProgressBar progress={
                {current: currentQuestion, ends: questions.length}
            } />
        </header>
    </div>
    <div class="question-wrapper">
        {#each questions as q, i}
            {#if i === currentQuestion}
                <h2>{q.question}</h2>
                {#if q?.subText}
                    <p>{q.subText}</p>
                {/if}
                <ul class="question-item-wrapper">
                    {#if q?.getOptions}
                        {#await q.getOptions() }
                            <p>Fetching resources</p>
                        {:then data}
                            {#each data as o}
                                <li class="item-wrapper">
                                    <button on:click={() => toggleAnswer(o.toLocaleLowerCase())}>
                                        {@html o}
                                    </button>
                                </li>
                            {/each}
                        {:catch error}
                            <p class="error-message">Couldn't fetch resources :(</p>
                            <p>Check if the session has expired.</p>
                        {/await}
                    {:else}
                        {#each q.options as o}
                            <button on:click={() => toggleAnswer(o.toLocaleLowerCase())}>
                                {o}
                            </button>
                        {/each}
                    {/if}
                </ul>
            {/if}
        {/each}

        <button disabled={currentQuestion > questions.length} on:click={progress}>
            Done
        </button>
    </div>
</div>

<style>
    .questionere-wrapper {
        margin-top: 2rem;
    }
    .header-wrapper {
        text-align: center;
    }
    .question-item-wrapper {
        height: 59vh;
        overflow-y: scroll;
    }
    .item-wrapper {
        display: flex;
        flex-direction: row;
    }
    ul {
        padding-left: 0;
    }
    div ul li {
        margin: 0.25em 0;
        text-align: left;
        width: 100%;
    }
    :global(.genre-wrapper ul) {
        list-style-type: none;
        padding-left: 0;
    }
    :global(.genre-wrapper ul li) {
        padding: 0.3em 0.5em;
        margin: 0.25em;
        background-color: rgba(0, 0, 0, 0.4);
        display: inline-block;
        border-radius: 0.25em;
        max-width: 35%;
    }
    :global(div ul li > *) {
        width: 100%;
        text-align: left;
    }
</style>
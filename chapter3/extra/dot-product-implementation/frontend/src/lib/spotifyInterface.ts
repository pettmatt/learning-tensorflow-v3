export interface Question {
    question: string
    subText?: string
    uri: string
    options: string[]
    getOptions?: Function
    answers: string[]
    followUp?: any[]
}

type Image = {
    height: number
    width: number
    url: string
}

interface DetailedArtist extends BaseArtist {
    genres: string[]
    followers: {
        href: string | null
        total: number
    }
    images: Image
    popularity: number
}

type BaseArtist = {
    external_urls: { spotify: string }
    href: string
    id: string
    name: string
    uri: string
    type: string
}

export interface Artist extends BaseArtist, DetailedArtist {}

type Album = {
    id: string
    href: string
    name: string
    type: string
    uri: string
    release_date: string
    release_date_precision: string
    total_tracks: number
    album_type: string
    artists: Artist[]
    available_markets: string[]
    external_urls: { spotify: string }
    images: Image[]
}

export interface Track {
    id: string
    name: string
    href: string
    popularity: number
    preview_url: string
    track_number: 6
    is_local: boolean
    album: Album
    artists: Artist[]
    disc_number: number
    duration_ms: number
    explicit: boolean
    external_ids: { isrc: string }
    external_urls: { spotify: string }
    type: string
    uri: string
}
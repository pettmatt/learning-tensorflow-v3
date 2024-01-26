export interface Question {
    question: string
    subText?: string
    uri: string
    options: string[]
    getOptions?: Function
    answers: string[]
    followUp?: any[]
}

export interface Item {
    external_urls: { spotify: string }
    genres: string[]
    followers: {
        href: string | null
        total: number
    }
    images: {
        height: number
        width: number
        url: string
    }
    href: string
    id: string
    name: string
    popularity: number
    uri: string
    type: string
}

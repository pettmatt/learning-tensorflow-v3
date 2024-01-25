export default async function pixelShift(input, mutations: any[] = []) {
    const padded = input.pad([[1, 1], [1, 1]], 1)
    const cutSize = input.shape

    for (let height = 0; height < 3; height++) {
        for (let width = 0; width < 3; width++) {
            mutations.push(padded.slice([height, width], cutSize))
        }
    }

    padded.dispose()
    return mutations
}
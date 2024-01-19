import * as dfd from "danfojs-node"
import optimizeDatasetBasedOnSex from "./optimize-model.js"

export async function fetchTitanicData() {
    try {
        // Check if clean-data directory has the files before creating them
        const dfTrain = await dfd.readCSV("./clean-data/cleanTest.csv") || null
        const dfTest = await dfd.readCSV("./clean-data/cleanTrain.csv") || null

        return [dfTest, dfTrain]
    } catch (error) {
        console.log("fetchData.js fetchTitanicData:", error)
        // Otherwise fetch and create datasets
        // If danfo.js is unable to fetch files, it will prompt an error
        const datasetSamples = createUsableTitanicDatasets()
        return datasetSamples
    }
}

async function createUsableTitanicDatasets() {
    const df = await dfd.readCSV("../../extra/titanic data/train.csv")
    const dft = await dfd.readCSV("../../extra/titanic data/test.csv")

    console.log("Train size", df.shape[0])
    console.log("Test size", dft.shape[0])

    // Combining file contents so the empty value check can be done once
    const full = dfd.concat({ dfList: [df, dft], axis: 0 })
    // full.describe().print()

    const cleanFullList = cleanDataset(full)
    const optimizeDataset = optimizeDatasetBasedOnSex(dfd, cleanFullList)
    const encodedFullList = encodeDatasetColumns(optimizeDataset, ["Embarked"])
    const separatedLists = createSampleDatasets(encodedFullList, true)
    return separatedLists
}

function cleanDataset(data) {
    console.log("Row-count", data.shape[0])
    const emptySpots = data.isNa().sum()
    const emptyRate = emptySpots.div(data.isNa().count())
    // console.log("Empty rate (Going to be cleaned)")
    // emptyRate.print()

    const clean = data.drop({
        columns: ["PassengerId", "Cabin", "Ticket"]
    })

    // DropNa() should make sure the end product is complete record of a passenger without missing data.
    const intactDataSet = clean.dropNa()
    console.log("After cleaning the row-count is now", intactDataSet.shape[0])
    
    return intactDataSet
}

export function encodeDatasetColumns(data, columnNames) {
    if (typeof columnNames !== "object") {
        console.log("EncodeDataseColumns-function: Please pass an array as the secondary argument.")
        return
    }

    columnNames.forEach(colN => {
        const encode = new dfd.LabelEncoder()
        const targetColumn = data[colN]
        encode.fit(targetColumn)
        data[colN] = encode.transform(targetColumn.values)
    })

    // Uncomment if visual check is necessary.
    // data.head().print()

    return data
}

async function createSampleDatasets(data, createSampleFiles = false) {
    let datasets = []

    // For some reason sample() breaks the application when trying to pass bigger numbers than 420.
    // This is caused because the data doesn't include over 420 rows (which is false, we have over 1000 records),
    // which is why we create test samples first and the remanining is used for training.
    // Ofcourse there is some unwanted results, which is the sample() shuffles the result, which could be useful for training dataset.
    const newTestingDataset = await data.sample(200)
    console.log("New testing dataset row count:", newTestingDataset.shape[0])
    datasets.push(newTestingDataset)

    const newTrainingDataset = data.drop({ index: newTestingDataset.index, axis: 0 })
    console.log("New training dataset row count:", newTrainingDataset.shape[0])
    datasets.push(newTrainingDataset)

    if (createSampleFiles) {
        createFile(datasets[0], { path: "./clean-data/cleanTest.csv" })
        createFile(datasets[1], { path: "./clean-data/cleanTrain.csv" })
    }

    return datasets
}

async function createFile(data, fileDetails) {
    try {
        dfd.toCSV(data, { filePath: fileDetails.path })
        console.log("File created:", fileDetails.path)
    } catch (error) {
        console.warn("FetchData.js, createFile-function. Be sure to create 'clean-data' directory in the root.", error)
    }
}
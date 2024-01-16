import { encodeDatasetColumns } from "./fetchData.js"

export default function optimizeDatasetBasedOnSex(dfd, data) {
    console.log("\n=========== Optimize dataset based on sex and honorific ===========\n")
    const sexOneHot = dfd.getDummies(data["Sex"], { prefix: "is" })
    sexOneHot.head().print()
    
    data.drop({ columns: ["Sex"], axis: 1, inplace: true })
    data.addColumn("Male", sexOneHot["is_male"], { inplace: true })
    data.addColumn("Female", sexOneHot["is_female"], { inplace: true })
    
    const ageBuckets = data["Age"].apply(ageToBucket) // apply function to every row of data
    data.addColumn("Age_bucket", ageBuckets, { inplace: true })
    
    // Because some people didn't survive based on their "title",
    // we can try to enhance our model by including the name honorific.
    data["Name"] = data["Name"].apply(nameHonorificToBucket)
    data.head().print()
    const encodedData = encodeDatasetColumns(data, ["Name"])

    const normalizedData = normalizeData(dfd, encodedData)
    return normalizedData
}

// Grouping in specific groups, like in age groups can  
// stream line the process of training the model.
// Which is also known as bucketing, spearating groups and putting them in "buckets".
// In case of Titanic there is somewhat clear grouping how many passengers survived by age.
function ageToBucket(age) {
    if (age < 10) return 0
    else if (age < 40) return 1
    else return 2
}

function nameHonorificToBucket(name) {
    return name.split(" ").filter(p => p.includes("."))[0].replace(".", "")
}

function normalizeData(dfd, data) {
    let scaler = new dfd.MinMaxScaler()
    scaler.fit(data)

    const dfEnc = scaler.transform(data)
    dfEnc.print()

    return dfEnc
}
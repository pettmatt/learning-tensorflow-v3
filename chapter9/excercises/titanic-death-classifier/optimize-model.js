import { encodeDatasetColumns } from "./fetchData.js"

export default function optimizeDatasetBasedOnSex(dfd, data) {
    console.log("\n=========== Optimize dataset based on sex and honorific ===========\n")

    // Because some people didn't survive based on their "title",
    // we can try to enhance our model by including the name honorific.
    // "apply"-function breaks the program when the "Name" column is being used,
    // creating following error: "File format not supported!"

    // Including the honorifics within their own column. Approach 1.
    // Produces: 15ms 94us/step - acc=0.825 loss=0.372 val_acc=0.825 val_loss=0.504
    // const names = data["Name"].values
    // const honorifics = names.map((name) => nameToHonorific(name))
    // data.addColumn("Name", honorifics, { inplace: true })
    data.head().print()

    // One-hot encoding removes the relation between the values.
    // Meaning the model is unable to compare the values. For example, in genders 
    // it could make a conclusion that one gender is better than the other.
    const sexOneHot = dfd.getDummies(data["Sex"], { prefix: "Is" })
    const nameOneHot = dfd.getDummies(data["Name"], { prefix: "Honorific" })
    sexOneHot.head().print()
    nameOneHot.head().print()

    data.drop({ columns: ["Sex"], axis: 1, inplace: true })
    data.addColumn("Male", sexOneHot["Is_male"], { inplace: true })
    data.addColumn("Female", sexOneHot["Is_female"], { inplace: true })

    // Part of approach 1.
    // data.drop({ columns: ["Name"], axis: 1, inplace: true })
    // let newData = dfd.concat({ dfList: [data, nameOneHot], axis: 1 })
    // newData.head().print()

    // Put honorifics within buckets. Approach 2.
    // Produces: (might be overfitted)
    // first run: 14ms 90us/step - acc=0.875 loss=0.294 val_acc=0.750 val_loss=0.672
    // second run: 56ms 84us/step - acc=0.862 loss=0.349 val_acc=0.976 val_loss=0.163
    // third run: 58ms 86us/step - acc=0.856 loss=0.350 val_acc=0.982 val_loss=0.184
    const honorifics = data["Name"].apply(nameToHonorific)
    data.addColumn("Honorific_bucket", honorifics, { inplace: true })
    data.drop({ columns: ["Name"], axis: 1, inplace: true })

    const ageBuckets = data["Age"].apply(ageToBucket) // apply function to every row of data
    data.addColumn("Age_bucket", ageBuckets, { inplace: true })

    const normalizedData = normalizeData(dfd, data)
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

function nameToHonorific(name) {
    const honorific = name.split(" ").filter(p =>
        p.includes("."))[0].replace(".", "").toLowerCase()

    if (honorific === "mr") return 0
    else if (honorific === "mrs") return 1
    else if (honorific === "miss") return 2
    else if (honorific === "master") return 3
    else if (honorific === "don") return 4
    else if (honorific === "rev") return 5
    else if (honorific === "dr") return 6
    else if (honorific === "mme") return 7
    else if (honorific === "ms") return 8
    else if (honorific === "major") return 9
    else if (honorific === "lady") return 10
    else if (honorific === "sir") return 11
    else if (honorific === "mlle") return 12
    else if (honorific === "col") return 13
    else if (honorific === "capt") return 14
    else if (honorific === "countess") return 15
    else if (honorific === "jonkheer") return 16
    else if (honorific === "dona") return 17

    console.log(`No honorific if-clause found for "${honorific}".`)
}

function normalizeData(dfd, data) {
    let scaler = new dfd.MinMaxScaler()
    scaler.fit(data)

    const dfEnc = scaler.transform(data)
    dfEnc.print()

    return dfEnc
}
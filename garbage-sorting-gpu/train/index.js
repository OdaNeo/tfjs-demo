const tf = require('@tensorflow/tfjs-node')
const getDate = require('./data')

const MOBILE_NET_URL =
    "https://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json"

// const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/new/model.json'

const main = async () => {
    // load data
    const { ds, classes } = await getDate()

    // define model
    const mobileNet = await tf.loadLayersModel(MOBILE_NET_URL)
    // const mobileNet = await tf.loadGraphModel(MOBILENET_MODEL_PATH, { fromTFHub: false })

    // new model
    const model = tf.sequential()

    for (let i = 0; i <= 86; i++) {
        const layer = mobileNet.layers[i]
        layer.trainable = false
        model.add(layer)
    }

    // flat layers
    model.add(tf.layers.flatten())

    model.add(tf.layers.dense({
        units: 20,
        activation: 'relu'
    }))
    model.add(tf.layers.dense({
        units: classes.length,
        activation: 'softmax'
    }))

    // train
    model.compile({
        loss: "sparseCategoricalCrossentropy",
        optimizer: tf.train.adam(),
        metrics: ["acc"],
    })
    try {
        await model.fitDataset(ds, {
            epochs: 20
        })
    } catch (err) {
        console.log(err)
    }


    // save
    await model.save(`file://${process.cwd()}/output`)
}

main()
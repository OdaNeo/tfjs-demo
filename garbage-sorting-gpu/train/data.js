const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')



const img2x = (imgPath) => {
    const buffer = fs.readFileSync(imgPath)
    return tf.tidy(() => {
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer))
        const imgTsResized = tf.image.resizeBilinear(imgTs, [224, 224])
        return imgTsResized.toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3])
    })
}
const getDate = async () => {

    const TRAIN_DIR = 'data/train'
    const classes = fs.readdirSync(TRAIN_DIR).filter(n => !n.includes('.'))
    fs.writeFileSync('output/classes.json', JSON.stringify(classes))

    //dataset
    const data = []

    classes.forEach((dir, dirIndex) => {
        fs.readdirSync(`${TRAIN_DIR}/${dir}`)
            .filter(n => n.match(/.jpg$/))
            // .slice(0, 100)
            .forEach(filename => {
                const imgPath = `${TRAIN_DIR}/${dir}/${filename}`
                data.push({ imgPath, dirIndex })
            })
    })

    tf.util.shuffle(data)

    const ds = tf.data.generator(function* () {
        const count = data.length;
        const batchSize = 100
        for (let start = 0; start < count; start += batchSize) {
            const end = Math.min(start + batchSize, count)
            yield tf.tidy(() => {
                const inputs = []
                const labels = []

                for (let j = start; j < end; j += 1) {
                    const { imgPath, dirIndex } = data[j]
                    const x = img2x(imgPath)
                    inputs.push(x)
                    labels.push(dirIndex)
                }
                const xs = tf.concat(inputs)
                const ys = tf.tensor(labels)
                return { xs, ys }
            })
        }
    })

    return {
        ds,
        classes
    }
}


module.exports = getDate
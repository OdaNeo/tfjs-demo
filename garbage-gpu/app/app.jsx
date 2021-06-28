import React, { useEffect } from "react"
import * as tf from '@tensorflow/tfjs'
import { file2img, img2tenser } from './utils'
import CLASSES from '../output/classes.json'

const DATA_URL = 'http://localhost:5000/model.json'

function  App (){

    useEffect(async()=>{
        this.model =await tf.loadLayersModel(DATA_URL)
    },[])

    const predict= async(file)=>{
        const img = await file2img(file)
      
        const pred = tf.tidy(()=>{
            const x=img2tenser(img)
            return this.model.predict(x)
        })

        const result = pred.arraySync()[0]
            .map((score ,i)=>({score,label:CLASSES[i]}))
            .sort((a,b)=>b.score-a.score)

        console.log(result[0].label)
    }

    return(
        <input onChange={e=>predict(e.target.files[0])} type='file'/>
    )
}

export default App
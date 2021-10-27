let model;
const webcam = new Webcam(document.getElementById('webcam'));
let isPredicting = false;

function getUserMediaSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function loadModel() {
    const trained_model = await tf.loadLayersModel('assets/model/model-modified.json'); // xception
    const layer = trained_model.getLayer('output');
    return tf.model({ inputs: trained_model.inputs, outputs: layer.output });
}

async function predict() {
    while (isPredicting) {
        var arr_score = [];
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const predictions = model.predict(img);
            arr_score = predictions.dataSync();
            return predictions.as1D().argMax();
        });
        const classId = (await predictedClass.data())[0];
        const score = arr_score[classId];
        var accText = (Math.round(score * 100000) / 1000).toString() + '%';
        const arr_classid = ["Hari ini", "Isyarat", "Kamu", "Maaf", "Makan", "Rumah", "Sakit", "Saya", "Teman", "Tidak", "OOV"];
        var predictionText = arr_classid[classId];
        console.log(accText);
        console.log(predictionText);
        
        if (predictionText == "OOV" || (predictionText != "OOV" && (Math.round(score * 100000) / 1000) < 80.0)) {
            accText = "";
            predictionText = "";
        }
        document.getElementById("prediction").innerText = predictionText;
        document.getElementById("accuracy").innerText = accText;

        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function startPredict() {
    isPredicting = true;
    predict();
}

function stopPredict() {
    isPredicting = false;
    predict();
}

async function init() {
    // const videoConstraints = {};
    // videoConstraints.deviceId = { exact: '46fdccbcd1f9c16ec5d6f4c944078b075ecea05f65b62ab5d57a76afc5b120a4' };
    // const constraints = {
    //     video: videoConstraints,
    //     audio: false
    // };
    // navigator.mediaDevices.getUserMedia(constraints).catch(error => {console.error(error);});
    console.log('[INFO] Starting to setup the webcam...');
    await webcam.setup();
    console.log('[INFO] Finish setup');
    model = await loadModel();
    tf.tidy(() => model.predict(webcam.capture()));
}

if (getUserMediaSupported()) {
    tf.setBackend('webgl');
    tf.ready().then(() => {
        console.log('WEBGL ready');
        init();
    });
} else {
    console.warn('getUserMedia() is not supported by your browser');
}
let video;
let features;
let knn;
let resultHtml;
let ready = false;
objects = [];
let label = '-';


const height = 700,
width = 950;

async function setup() {
  fetch('./objects-1.json').then(res => res.json()).then(res => {
    objects = res.objects;
    drawObjects();
  })
  createCanvas(width, height);
  video = createCapture(VIDEO);
  video.hide();
  features = ml5.featureExtractor('MobileNet', modelReady);
  knn = ml5.KNNClassifier();
  resultHtml = createP('Modelleri eğitin !');
  resultHtml.class('resultLabel')
}

function goClassify() {
  const logits = features.infer(video);
  knn.classify(logits, function(error, result) {
    if (error) {
      console.error(error);
    } else {
      label = result.label;
      resultHtml.html(result.label);
      goClassify();
    }
  });
}

function keyPressed() {
  const logits = features.infer(video);
  objects.forEach(object => {
    if (key == object.keypress.toString()) {
      knn.addExample(logits, object.name.toString());
    }
    else if (key == 's') {
      knn.save();
    }
  })
}

function modelReady() {
  console.log('model ready!');
  //  modelinizi iyice eğitinize emin olduktan sonra modelinizi yükleyin, ve model eğitimini durdurun.
  // knn.load('model.json', function() {
  //   console.log('knn loaded');
  // });
}

function drawObjects() {
  const objectsDiv = createDiv();
  objectsDiv.class('objectsInfo');
  objects.forEach(object => {
    const oEl = createP(`${object.name} objesini eğitmek için ${object.keypress} tuşuna basarak örnek ekleyiniz`);
    objectsDiv.child(oEl);
  })
  const saveEl = createP();
  saveEl.html(`Modeli kaydetmek için <b>S</b> tuşuna basınız`)
  objectsDiv.child(saveEl);
}

function draw() {
  image(video, 0, 0 , width, height);
  if (!ready && knn.getNumLabels() > 0) {
    goClassify();
    ready = true;
  }
}

from django.shortcuts import render
import tensorflow as tf
import numpy as np
import cv2

FLOWER_LABELS = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

def classify(request):

    label = False
    fileName = False

    if request.method == "POST" and request.FILES.get('flower_image', False):

        fileName = request.FILES['flower_image'].name
        flower_image = request.FILES["flower_image"].read()

        # process/decode/format the image 
        flower_image = cv2.imdecode(np.frombuffer(flower_image, np.uint8), cv2.IMREAD_COLOR)
        flower_image = cv2.resize(flower_image, (180, 180))
        flower_image = np.expand_dims(flower_image, axis=0)

        model = tf.keras.models.load_model("models/base_model.h5")
        predict = model.predict(flower_image)
        prediction = np.argmax(predict)

        label = FLOWER_LABELS[prediction]

    return render(request, 'ml_model/classify.html', { 'label': label, 'fileName': fileName })

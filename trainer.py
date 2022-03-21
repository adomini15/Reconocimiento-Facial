import os
import numpy as np
from PIL import Image
import cv2
import pickle

# clasificador
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')

# reconocedor de cara
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Obtendremos el directorio donde esta este script

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(BASE_DIR, 'images') # {BASE_DIR}/images

generated_id = 0
label_ids = {}
labels = []
train = []

for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()

            # cada nombre tendra asignado un id numerico
            # esta info lo guardamos en un diccionario
            if label not in label_ids:
                label_ids[label] = generated_id
                generated_id+=1

            current_person_id = label_ids[label]

            pil_image = Image.open(path).convert('L') # convierte en escala de grises
            image_array = np.array(pil_image, 'uint8')
            # el nuevo resultado aún mantendrá la info. sobre escala de grises,
            # transformación requerida para el detectMultiScale

            # Toma cada pixel de la imagen y las coloca dentro de un array de numpy,
            # representados como valores enteros positivos (de pixel binario a 'uint8')

            # PROCESO DE RECONOCIMIENTO

            # Esta operación podría ser menos pesada que en capture, debido a que lo que
            # proyecta la imagen_array es una cara anteriormente guardada.
            faces = face_cascade.detectMultiScale(image_array)

            for (x,y, w, h) in faces:
                roi = image_array[y: y+h, x: x+w] # region of interest
                train.append(roi)
                labels.append(current_person_id)

with open('pickles/labels.pkl', 'wb') as file:
    pickle.dump(label_ids, file)

recognizer.train(train, np.array(labels))
recognizer.save('recognizers/faces-trainer.yml')
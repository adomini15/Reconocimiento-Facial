import numpy as np
import cv2
import os

# Arriba se importa el paquete de opencv instalado
# y tambien numpy, utilizado dentro del modulo cv2

# Debajo preparamos un modelo de entrenamiento que luego utilizaremos
# ofrecido por opencv en haar cascades, para frontalface detection
# clasificador:
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')

# capturadora de video operando en background.
capture = cv2.VideoCapture(0)

# bucle infinito, utilizado para ir
# leyendo e imprimiendo en una ventana el frame actual obtenido del 'capture'
name = input('Ingrese su nombre:')
generated_num = 0

while True and name:
    ret, frame = capture.read()

    # Recluimos de la etapa debajo, de  manipulación de la imagen,
    # después de la etapa de extracción (véase etapas del reconocimiento de patrones),
    # en donde convertiremos nuestra imagen o frame de video en escala de grises.
    # Necesario para el clasificador.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detección de cara en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # De cada item tupla que tiene faces, hacemos una asignación destructurante
    # para tratar los pixeles almacenar el punto x, el y, el with y el hight,
    # en x, y, w, h, respectivamente.
    # ESOS VALORES LO VAMOS A UTILIZAR PARA 'RECORTAR' LA REGION QUE NOS INTERESA DE 'gray',
    # QUE ESTARIA CORRESPONDIENTE AL AREA DETECTADA COMO CARA EN CADA ITEM DE 'faces', ;).
    for (x,y,w,h) in faces:
        # utilizamos el tipico range de los valores secuencias de python
        # para leer dentro de las coordenas (x, w + x), y (y,h + y). :).
        roi_gray = frame[y:y+h, x:x+w] # roi: se entiende por región de interes.
        # Utilizamos frame en lugar gray, porque lo queremos de la imagen original a color.

        dirname = name

        if not os.path.exists(os.path.join('images', dirname)):
            os.mkdir(os.path.join('images', dirname))

        # vamos a guardar localmente la imagen en 'roi_gray' obtenida.
        img_item = str(generated_num) + ".png"  # nombre que tendrá el archivo
        generated_num = 1 + generated_num

        cv2.imwrite(os.path.join('images', dirname, img_item), roi_gray)

        # vamos a ir dibujando un rectangulo reutilizando las coordenadas
        # que estamos utilizando. Todo eso lo hacemos despues de haber guardado.
        # No queremos que en el archivo aparezca ese rectangulo.
        # Solo es para saber donde esta nuestra cara cuando hacemos imshow de 'frame'

        # establecemos sus estilos
        color = (255,140,45) # NOTE: Aquí no lo utilizamos como RGB, si no BGR. :|.
        # ancho de borde
        stroke = 5
        # ancho
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    # impresión del frame actual de la iteración en una ventana
    cv2.imshow('Frame', frame)

    # condición establecida para poder finalizar
    # el bucle cuando haya sido presionada la letra 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# eliminamos el recurso de videocaptura ejecutando en background.
capture.release()

# Cerramos todas las ventanas creadas por cv2 abiertas.
cv2.destroyAllWindows()
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Carga el modelo de la carpeta 'Reconocimientotexto'
model = load_model('Reconocimientotexto/keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
image = Image.open("Reconocimientotexto/deteccion.png")
st.image(image, caption="detección")

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

# Permite tomar una foto desde la cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # Convierte el archivo de imagen a un formato compatible con PIL
    img = Image.open(img_file_buffer)

    # Redimensiona la imagen a 224x224
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convierte la imagen PIL a un array de numpy
    img_array = np.array(img)

    # Normaliza la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Carga la imagen en el array
    data[0] = normalized_image_array

    # Ejecuta la inferencia con el modelo
    prediction = model.predict(data)

    # Muestra el resultado de la predicción
    if prediction[0][0] > 0.5:
        st.header('Izquierda, con Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('Arriba, con Probabilidad: ' + str(prediction[0][1]))
    # Si deseas agregar más clases, puedes descomentar y ajustar la siguiente línea:
    # if prediction[0][2] > 0.5:
    #     st.header('Derecha, con Probabilidad: ' + str(prediction[0][2]))

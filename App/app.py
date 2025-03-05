import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image


@st.cache_resource
def load_trained_model():
    model_path = r"C:\Users\aayus\Documents\Brahmastra\Kingdom of Heaven\Tester\my_model.h5"
    return load_model(model_path)

model = load_trained_model()


class_labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


img_size = (224, 224)

st.title("Diabetic Retinopathy Prediction - Upload Images")


uploaded_files = st.file_uploader("Upload Eye Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
     
        img = image.resize(img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

     
        input1 = img_array
        input2 = img_array

        prediction = model.predict([input1, input2])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        st.success(f"### Prediction: {predicted_label}")

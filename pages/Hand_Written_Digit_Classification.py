import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/mnist_model.h5")

st.title("Handwritten Digit Classification")

uploaded_file = st.file_uploader("Upload digit image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))
    st.image(image, width=150)

    img = np.array(image)/255.0
    img = img.reshape(1,28,28)

    pred = model.predict(img)
    digit = np.argmax(pred)

    st.success(f"Predicted Digit: {digit}")

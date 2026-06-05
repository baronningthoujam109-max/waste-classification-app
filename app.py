import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------- PAGE -----------------
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="centered"
)

# ----------------- MODEL -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier.keras")

model = load_model()

# ----------------- CLASS NAMES -----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------- UI -----------------
st.title("♻️ Waste Classification AI")
st.write("Upload an image to classify waste type.")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

# ----------------- MODE SWITCH (IMPORTANT DEBUG TOOL) -----------------
mode = st.radio(
    "Choose preprocessing mode",
    ["EfficientNet (recommended)", "Simple (debug)"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)

    # ----------------- PREPROCESSING -----------------

    if mode == "EfficientNet (recommended)":
        img = preprocess_input(img)
    else:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # ----------------- PREDICTION -----------------
    prediction = model.predict(img, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # ----------------- OUTPUT -----------------
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # ----------------- DEBUG -----------------
    with st.expander("🔍 Debug Info"):
        st.write("Prediction vector:", prediction)
        st.write("Argmax:", predicted_index)
        st.write("Class names:", class_names)
        st.write("Sum:", np.sum(prediction))

        st.bar_chart(prediction[0])
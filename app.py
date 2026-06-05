import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="centered"
)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier.keras")

model = load_model()

# ----------------- CLASS NAMES -----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------- UI -----------------
st.title("♻️ Waste Classification AI")
st.write("Upload an image and get prediction.")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

# ----------------- PREDICTION -----------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)

    # MUST match training
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
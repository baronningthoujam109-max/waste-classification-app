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

if uploaded_file:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Convert to array
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert to float32
    img = img.astype(np.float32)

    # 🔥 IMPORTANT FIX (THIS IS THE KEY)
    img = preprocess_input(img)

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    with st.expander("Debug"):
        st.write(prediction)
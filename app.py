import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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
st.write("Upload an image and get waste classification result.")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

# ----------------- PREDICTION -----------------
if uploaded_file:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert to float
    img = img.astype(np.float32)

    # 🔥 IMPORTANT: must match training
    img = preprocess_input(img)

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Predict
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
        st.write("Predicted index:", predicted_index)
        st.write("Class names:", class_names)
        st.write("Sum of probabilities:", np.sum(prediction))

        # Safe visualization (NO Altair error)
        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0])
        plt.xticks(rotation=45)
        st.pyplot(fig)
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2

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

# ----------------- LOAD CLASS NAMES -----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------- UI -----------------
st.title("♻️ Waste Classification AI")
st.write("Upload an image and the model will predict the waste category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# ----------------- PREDICTION -----------------
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy
    img = np.array(image)

    # Resize first (IMPORTANT)
    img = cv2.resize(img, (224, 224))

    # OPTIONAL: light denoising (safe)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Convert to float
    img = img.astype(np.float32)

    # ⚠️ IMPORTANT:
    # Try WITHOUT normalization first (most models trained this way in projects)
    # If accuracy is low, we can enable /255.0 later
    # img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # ----------------- OUTPUT -----------------
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Debug section (helpful)
    with st.expander("🔍 Debug Info"):
        st.write("Raw prediction:", prediction)
        st.write("Class index:", predicted_index)
        st.write("Class names:", class_names)
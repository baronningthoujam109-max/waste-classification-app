import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Waste Classification by Baron_Ningthoujam",
    page_icon="♻️",
    layout="centered"
)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb0_waste.keras")

model = load_model()

# ----------------- LOAD CLASS NAMES -----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------- UI -----------------
st.title("♻️ Waste Classification by Baron_Ningthoujam")
st.write("Upload an image to classify waste type.")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

# ----------------- PREDICTION -----------------
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # ----------------- OUTPUT -----------------
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # ----------------- PROBABILITY TABLE -----------------
    probs = prediction[0]
    sorted_idx = np.argsort(probs)[::-1]

    prob_table = {
        "Class": [class_names[i] for i in sorted_idx],
        "Probability (%)": np.round(probs[sorted_idx] * 100, 2)
    }

    st.subheader("📊 Class Probabilities")
    st.table(prob_table)

    # ----------------- SOURCE INFO -----------------
    st.divider()

    st.subheader("🔗 Source & Model Info")

    st.markdown(
        """
        📂 **Project Source Code:**  
        https://github.com/baronningthoujam109-max/waste-classification-app  

        🧠 **Model:** EfficientNetB0 + TensorFlow  
        ♻️ **Task:** Waste Classification (Glass, Metal, Paper, Plastic, Cardboard, Trash)
        """
    )

    st.caption("Built using TensorFlow + EfficientNetB0 + Streamlit")
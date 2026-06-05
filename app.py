import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2

from PIL import Image

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_classifier.keras"
    )

model = load_model()

with open("class_names.json","r") as f:
    class_names = json.load(f)

st.title("♻️ Waste Classification AI")

st.write(
    "Upload a waste image and the model will predict its category."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = np.array(image)

    img = cv2.GaussianBlur(
        img,
        (3,3),
        0
    )

    img = cv2.resize(img, (224, 224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)
img = img.astype("float32") / 255.0

    prediction = model.predict(img)

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = (
        np.max(prediction) * 100
    )

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
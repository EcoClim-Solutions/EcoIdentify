# Your enhanced app code
import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *
import gdown
import requests
import tensorflow as tf

# Set page title and favicon
st.set_page_config(page_title="Garbage Segregation App", page_icon="🗑️")

# Set style for the app
st.markdown(
    """
    <style>
        body {
            color: #1E1E1E;
            background-color: #F8F8F8;
        }
        .st-bb {
            padding: 0rem;
        }
        .st-ec {
            color: #6E6E6E;
        }
        .st-ef {
            color: #6E6E6E;
        }
        .st-ei {
            color: #1E1E1E;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("Garbage Segregation")

# Subheader
st.subheader("Upload an image to classify its waste category")

# Selectbox for upload option
opt = st.selectbox(
    "How do you want to upload the image for classification?",
    ("Please Select", "Upload image via link", "Upload image from device"),
)

# Initialize variables
image = None
model = None

# Upload image based on user selection
if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    if file is not None:
        image = Image.open(file).resize((256, 256), Image.LANCZOS)

elif opt == "Upload image via link":
    try:
        img = st.text_input("Enter the Image Address")
        image = Image.open(urllib.request.urlopen(img)).resize((256, 256), Image.LANCZOS)
    except:
        if st.button("Submit"):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

# Display uploaded image
if image is not None:
    st.image(image, width=256, caption="Uploaded Image")

    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = preprocess(image)
            model = model_arc()
            prediction = model.predict(np.expand_dims(img, axis=0))

        # Display top predictions
        top_n = 3  # Display top 3 predictions
        top_classes = np.argsort(prediction[0])[::-1][:top_n]
        for i, class_idx in enumerate(top_classes):
            st.success(f"Prediction {i+1}: {labels[class_idx]} with confidence {prediction[0][class_idx]:.2%}")

# Clear Button
if st.button("Clear"):
    image = None
    st.image(image, width=256, caption="Uploaded Image")

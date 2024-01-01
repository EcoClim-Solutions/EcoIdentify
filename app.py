import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import preprocess, model_arc
import gdown
import requests
import tensorflow as tf

model = model_arc()

# Set page title and favicon
st.set_page_config(page_title="Garbage Segregation App", page_icon="https://ecoclimsolutions.files.wordpress.com/2023/11/ecoclim-logo.png")

# Set style for the app
st.markdown(
    """
    <style>
        body {
            color: #FFFFFF;
            background-color: #3498db;
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
labels = ['cardboard', 'glass', 'paper', 'plastic', 'metal', 'trash']

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
            prediction = model.predict(img)
            print(f"Debug - Predictions: {prediction}")

        
        # Display top prediction
        topclass = np.argmax(prediction)

        # Make sure top_class is an integer
        top_class = int(topclass)


        confidence = prediction[0][top_class]

        # Provide an additional prediction
        second_prediction_idx = np.argsort(prediction[0])[::-1][1]
        second_confidence = prediction[0][second_prediction_idx]
        st.success(f"Prediction: {labels[top_class]} with confidence {confidence:.2%}")
        st.warning(f"Alternative Prediction: {labels[second_prediction_idx]} with confidence {second_confidence:.2%}")

# Clear Button
if st.button("Clear"):
    image = None
    st.image(image, width=256, caption="Uploaded Image")
    st.warning("Image cleared. Upload a new image for prediction.")

import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import preprocess, model_arc


# Set Streamlit page configuration
st.set_page_config(
    page_title="Garbage Segregation App",
    page_icon="https://ecoclimsolutions.files.wordpress.com/2023/11/ecoclim-logo.png"
)

# Define class labels
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Set custom styles
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
        .st-ec, .st-ef {
            color: #6E6E6E;
        }
        .st-ei {
            color: #1E1E1E;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main application logic
st.title("Garbage Segregation")
st.subheader("Upload an image to classify its waste category")

# Image upload options
opt = st.selectbox(
    "How do you want to upload the image for classification?",
    ("Please Select", "Upload image via link", "Upload image from device"),
)

image = None
prediction = None

if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).resize((256, 256), Image.LANCZOS)

elif opt == "Upload image via link":
    img_url = st.text_input("Enter the Image Address")
    if st.button("Submit"):
        try:
            image = Image.open(urllib.request.urlopen(img_url)).resize((256, 256), Image.LANCZOS)
        except:
            st.error("Please Enter a valid Image Address!")

if image:
    st.image(image, width=256, caption="Uploaded Image")

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img_array = preprocess(image)
            prediction = model.predict(img_array)
        
        top_class_idx = np.argmax(prediction)
        top_class = labels[top_class_idx]
        confidence = prediction[0][top_class_idx]

        second_prediction_idx = np.argsort(prediction[0])[::-1][1]
        second_confidence = prediction[0][second_prediction_idx]

        st.success(f"Prediction: {top_class} with confidence {confidence:.2%}")
        st.warning(f"Alternative Prediction: {labels[second_prediction_idx]} with confidence {second_confidence:.2%}")

if st.button("Clear"):
    image = None
    st.warning("Image cleared. Upload a new image for prediction.")

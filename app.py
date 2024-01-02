import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import tensorflow as tf
from utils import preprocess, model_arc, predict_image
from pathlib import Path
import gdown
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

urlforinstall = 'https://www.dropbox.com/scl/fi/8lxjfo0ebfd7kgb0sito6/EcoIdentify_official_classification_model.h5?rlkey=35jdpwthtr4fbfehz02abozf5&dl=1'
outputforinstall = 'EcoIdentify_official_classification_model.h5'
gdown.download(urlforinstall, outputforinstall, quiet=False)
model = tf.keras.models.load_model('EcoIdentify_official_classification_model.h5')

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
            example_image = transformations(image)
            plt.imshow(example_image.permute(1, 2, 0))
            st.success("The image resembles", predict_image(example_image, model) + ".")
        

        
if st.button("Clear"):
    image = None
    st.warning("Image cleared. Upload a new image for prediction.")


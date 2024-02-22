import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from utils import preprocess_image, image_details



# Initialize labels and model
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = tf.keras.models.load_model('classify_model.h5')

# Customized Streamlit layout
st.set_page_config(
    page_title="EcoIdentify by EcoClim Solutions",
    page_icon="https://ecoclimsolutions.files.wordpress.com/2024/01/rmcai-removebg.png?resize=48%2C48",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Customized Streamlit styles
st.markdown(
    """
    <style>
        body {
            color: #333333;
            background-color: #f9f9f9;
            font-family: 'Helvetica', sans-serif;
        }
        .st-bb {
            padding: 0rem;
        }
        .st-ec {
            color: #666666;
        }
        .st-ef {
            color: #666666;
        }
        .st-ei {
            color: #333333;
        }
        .st-dh {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .st-gf {
            background-color: #4CAF50;
            color: white;
            padding: 15px 30px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .st-gf:hover {
            background-color: #45a049;
        }
        .st-gh {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .st-logo {
            max-width: 100%;
            height: auto;
            margin: 20px auto;
            display: block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo
st.image("https://ecoclimsolutions.files.wordpress.com/2024/01/rmcai-removebg.png?resize=48%2C48")

# Page title
st.title("EcoIdentify by EcoClim Solutions")

# Subheader
st.header("Upload a waste image to find its category")

# Note
st.markdown("* Please note that our dataset is trained primarily with images that contain a white background.  Therefore, images with white background would produce maximum accuracy *")

# Image upload section
opt = st.selectbox("How do you want to upload the image for classification?", ("Please Select", "Upload image from device"))

image = None

if opt == 'Upload image from device':
    file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    if file:
        image = preprocess_image(file)

try:
    if image is not None:
        st.image(image, width=256, caption='Uploaded Image')
        st.write(image_details(file))
        if st.button('Predict'):
            prediction = model.predict(image[np.newaxis, ...])
            st.success(f'Prediction: {labels[np.argmax(prediction[0], axis=-1)]}')
except Exception as e:
    st.error(f"An error occurred: {e}.  Please contact us EcoClim Solutions at EcoClimSolutions.wordpress.com.")

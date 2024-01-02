import streamlit as st
from utils import predict
from Downloading_model import main
import tensorflow as tf
from PIL import Image
import urllib.request


# Set Streamlit page configuration
st.set_page_config(
    page_title="Garbage Segregation App",
    page_icon="https://ecoclimsolutions.files.wordpress.com/2023/11/ecoclim-logo.png",
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
            top_class, confidence = predict(image)

        st.success(f"Prediction: {top_class} with confidence {confidence:.2%}")

if st.button("Clear"):
    image = None
    st.info("Image cleared. Upload a new image for prediction.")


# Load the model only once
model_path = main()
model = tf.keras.models.load_model(model_path)

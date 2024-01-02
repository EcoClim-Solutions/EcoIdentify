import streamlit as st
from PIL import Image
import urllib.request
from utils import preprocess, predict_image
from Downloading_model import model_download


model = model_download("https://onedrive.live.com/download?resid=657A29EC827C9C58%21107&authkey=!APDOTvOiL9qk5wc")

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

if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    if file:
        processed_image = preprocess(file)

elif opt == "Upload image via link":
    img_url = st.text_input("Enter the Image Address")
    if st.button("Submit"):
        try:
            file = Image.open(urllib.request.urlopen(img_url))
            processed_image = preprocess(file)
        except:
            st.error("Please Enter a valid Image Address!")


if st.button("Predict"):
    with st.spinner("Predicting..."):
        class_label, prediction_shape = predict_image(processed_image)

        # Display the results
        print(f"The image resembles {class_label}. Prediction shape: {prediction_shape}.")
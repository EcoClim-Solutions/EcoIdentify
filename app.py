import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import gdown

# Download the model
model_path = 'https://www.dropbox.com/scl/fi/fzhpuhrqviqzbrcyvplyo/EcoIdentify_official_classification_model.h5?rlkey=bc9rm4e4fdfv7pxqxundiyxfp&dl=0'  # Replace with the actual URL
output_path = 'EcoIdentify_official_classification_model.h5'
gdown.download(model_path, output_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(output_path)

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to classify the garbage
def classify_garbage(img, model):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)

    class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    predicted_class = np.argmax(prediction, axis=1)[0]
    classification_result = class_labels[predicted_class]

    # Get the confidence (probability) of the predicted class
    confidence = prediction[0][predicted_class] * 100  # Convert probability to percentage

    return classification_result, confidence

# Streamlit app
st.title("Garbage Classifier")

# Function to take a photo using the webcam
def take_photo():
    st.write("Click the button below to take a photo:")
    button_pressed = st.button("Capture Photo")

    if button_pressed:
        st.info("Capturing photo... Please wait.")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Photo", use_column_width=True)
            return image

# Capture and classify
try:
    captured_photo = take_photo()
    
    if captured_photo:
        st.success('Photo captured successfully!')
        classification_result, confidence = classify_garbage(captured_photo, model)
        st.write(f"The item in the photo is: **{classification_result}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

except Exception as e:
    st.error(f"An error occurred: {e}")

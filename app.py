import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import requests

# Download from Dropbox function
def download_from_dropbox(url, save_path):
    response = requests.get(url, allow_redirects=True)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Model Dropbox link
MODEL_URL = "https://www.dropbox.com/scl/fi/87hcb77kbb0ygtjmiyfel?dl=1"  # Modified link

# Download and save the model
model_path = "model_directory"
download_from_dropbox(MODEL_URL, model_path)

# Load the model using TensorFlow
model = tf.saved_model.load(model_path)

def preprocess_image(img_str):
    encoded_img = img_str.split(';base64,')[-1]
    img = Image.open(BytesIO(base64.b64decode(encoded_img)))
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_garbage(img_array):
    prediction = model.predict(img_array)

    class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    predicted_class = np.argmax(prediction, axis=1)[0]
    classification_result = class_labels[predicted_class]

    confidence = prediction[0][predicted_class] * 100
    return classification_result, confidence

def image_mod(image):
    preprocessed_image = preprocess_image(image)
    classification_result, confidence = classify_garbage(preprocessed_image)
    return classification_result, confidence

demo = gr.Interface(
    image_mod,
    [gr.Image(type="pil")],
    [gr.Label("classification_result"), gr.Textbox("confidence")],
    flagging_options=["blurry", "incorrect", "other"],
    live=True
)

if __name__ == "__main__":
    demo.launch()
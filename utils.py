import gdown
import tensorflow as tf
from PIL import Image
import numpy as np
import Downloading_model

model = tf.keras.models.load_model(Downloading_model.main)

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def model_download():
    urlforinstall = 'https://www.dropbox.com/scl/fi/8lxjfo0ebfd7kgb0sito6/EcoIdentify_official_classification_model.h5?rlkey=35jdpwthtr4fbfehz02abozf5&dl=1'
    outputforinstall = 'EcoIdentify_official_classification_model.h5'

    gdown.download(urlforinstall, outputforinstall, quiet=False)

    # Return the path where the model is saved
    return outputforinstall


def preprocess(image):
    # Resize the image
    image = image.resize((256, 256), Image.LANCZOS)

    # Convert the image to a NumPy array and scale its values to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Add an extra dimension for batch size
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict(image):
    # Preprocess the image
    image_array = preprocess(image)

    # Make a prediction
    prediction = model.predict(image_array)

    # Get the index of the class with the highest probability
    top_class_idx = np.argmax(prediction)

    # Get the class name and confidence
    top_class = labels[top_class_idx]
    confidence = prediction[0][top_class_idx]

    return top_class, confidence

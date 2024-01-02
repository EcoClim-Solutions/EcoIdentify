#Imports
import numpy as np
from PIL import Image
import torch
from Downloading_model import model_download
from io import BytesIO




def preprocess(file):
    # Convert UploadedFile to PIL Image
    image = Image.open(file)

    # Resize the image
    image = image.resize((256, 256), Image.LANCZOS)

    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Normalize pixel values to be between 0 and 1
    image_array = image_array / 255.0

    # Return the processed image
    return image_array


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def predict_image(img, model, labels):
    
    imag = preprocess(img)  # Apply preprocessing specific to ResNet50

    # Make the prediction
    preds = model.predict(imag)

    # Process the prediction
    class_idx = preds.argmax(axis=-1)[0]
    class_label = labels[class_idx]
    prediction_shape = preds.shape

    return class_label, prediction_shape
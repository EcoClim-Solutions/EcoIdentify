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
    # Convert NumPy array to PyTorch tensor
    xb = torch.from_numpy(img).unsqueeze(0)

    # Assuming 'model' is your PyTorch model
    model.eval()

    # Make the prediction
    with torch.no_grad():
        preds = model(xb)

    # Process the prediction
    class_idx = torch.argmax(preds[0]).item()
    class_label = labels[class_idx]
    prediction_shape = preds.shape

    return class_label, prediction_shape

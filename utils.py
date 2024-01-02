from PIL import Image
import numpy as np
import torch

def preprocess(input_data):
    if isinstance(input_data, np.ndarray):
        # If input is a NumPy array, directly use it
        image = Image.fromarray((input_data * 255).astype(np.uint8))
    else:
        # If input is a file (e.g., uploaded file), open it
        image = Image.open(input_data)

    # Resize the image
    image = image.resize((256, 256), Image.LANCZOS)

    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)

    # Normalize pixel values to be between 0 and 1
    image_array = image_array / 255.0

    # Return the processed image
    return image_array

def predict_image(img, model, labels):
    # Convert NumPy array to PyTorch tensor
    xb = torch.from_numpy(img).unsqueeze(0)

    # Make the prediction
    preds = model(xb)

    # Process the prediction
    class_idx = torch.argmax(preds[0]).item()
    class_label = labels[class_idx]
    prediction_shape = preds.shape

    return class_label, prediction_shape

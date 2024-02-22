from PIL import Image
import numpy as np

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img_array = np.array(img)
    return img_array

# Function to classify the garbage
def classify_garbage(img_path, model):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)

    class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    predicted_class = np.argmax(prediction, axis=1)[0]
    classification_result = class_labels[predicted_class]

    # Get the confidence (probability) of the predicted class
    confidence = prediction[0][predicted_class] * 100  # Convert probability to percentage

    return classification_result, confidence

def image_details(img_path):
    details = {
    "Image name": image.name,
    "Image type": image.type,
    "Image size": image.size
    }
    return details

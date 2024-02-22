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
    
def image_details(uploaded_file):
    image = Image.open(uploaded_file)
    details = f"""
    - **Image Format**: {image.format}
    - **Image Mode**: {image.mode}
    - **Image Size**: {image.size[0]} pixels (width) x {image.size[1]} pixels (height)
    """
    return details




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import tensorflow as tf
from Downloading_model import model_download  # Assuming model_download is correctly implemented

def gen_labels():
    train = 'Dataset/Train'
    train_generator = ImageDataGenerator(rescale=1/255)

    train_generator = train_generator.flow_from_directory(train,
                                                        target_size=(300, 300),
                                                        batch_size=32,
                                                        class_mode='sparse')
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())

    return labels

def preprocess(image):
    # Resize the image
    image = image.resize((256, 256), Image.LANCZOS)
    
    # Convert the image to a NumPy array and scale its values to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add an extra dimension for batch size
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def model_arc():
    return model

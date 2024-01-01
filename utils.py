from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import tensorflow as tf

def gen_labels():
    train = 'Dataset/Train'
    train_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = train_generator.flow_from_directory(train,
                                                        target_size = (300,300),
                                                        batch_size = 32,
                                                        class_mode = 'sparse')
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    return labels

def preprocess(image):
    image = np.array(image.resize((256, 256), Image.LANCZOS))
    image = np.array(image, dtype='uint8')
    image = np.array(image) / 255.0

    return image

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
], name='data_augmentation')

#Instantiating the base model
input_shape = (256,256,3)
base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape)

#Making the layers of the model trainable
base_model.trainable = True

def model_arc():
    model = model = tf.keras.models.load_model('EcoIdentify_modellink.h5')
    return model

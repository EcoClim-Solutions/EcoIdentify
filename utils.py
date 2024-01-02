from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import tensorflow as tf
from Downloading_model import model_download  # Assuming model_download is correctly implemented
import torchvision as torch
from torchvision.datasets import ImageFolder
import gdown
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import shape

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = ImageFolder('Dataset', transform = transformations)


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
    model_path = model_download()
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with the highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()], yb.shape[1:]


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
"""device = get_default_device()
device

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    #"""Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
      #  """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
       # """Number of batches"""
        return len(self.dl)"""

#Imports
import numpy as np
from PIL import Image
import torch
from Downloading_model import model_download

#Preprocessing Images
def preprocess(image):
    # Resize the image
    image = image.resize((256, 256), Image.LANCZOS)
    
    # Convert the image to a NumPy array and scale its values to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add an extra dimension for batch size
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = get_default_device()
device

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
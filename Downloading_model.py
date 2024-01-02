from utils import model_download
import tensorflow as tf

def main():
    model_path = model_download()
    model = tf.keras.models.load_model(model_path)
    return model_path

import gdown
import tensorflow as tf

def model_download(path):
    urlforinstall = path
    outputforinstall = 'EcoIdentify_official_classification_model.h5'
    gdown.download(urlforinstall, outputforinstall, quiet=False)
    model = tf.keras.models.load_model(outputforinstall)
    return model


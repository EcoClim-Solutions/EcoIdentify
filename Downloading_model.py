import gdown

def model_download():
    urlforinstall = 'https://www.dropbox.com/scl/fi/8lxjfo0ebfd7kgb0sito6/EcoIdentify_official_classification_model.h5?rlkey=35jdpwthtr4fbfehz02abozf5&dl=1'
    outputforinstall = 'EcoIdentify_official_classification_model.h5'

    gdown.download(urlforinstall, outputforinstall, quiet=False)

    # Return the path where the model is saved
    return outputforinstall

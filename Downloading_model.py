import gdown

def model_download():
    urlforinstall = 'https://www.dropbox.com/scl/fi/uss7jgn63bfmz9p47y3es/EcoIdentify_modellink.h5?rlkey=metmtzv4tmnntl63upgu0px25&dl=1'
    outputforinstall = 'EcoIdentify_modellink.h5'

    gdown.download(urlforinstall, outputforinstall, quiet=False)

    # Return the path where the model is saved
    return outputforinstall

import gdown

def model_download():
    urlforinstall = 'https://drive.google.com/file/d/19cHovYhc-fiTCtQ2aXNlB5mozwTdKscx/view?usp=drive_link'
    outputforinstall = 'EcoIdentify_modellink.h5'

    gdown.download(urlforinstall, outputforinstall, quiet=False)
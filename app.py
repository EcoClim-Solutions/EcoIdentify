import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *
import requests

labels = gen_labels()

html_temp = '''
    <div style="padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Garbage Segregation</h1></center>
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)

html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload Waste Image to find its Category</h3></center>
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)

opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))

image = None  # Initialize image variable

if opt == 'Upload image from device':
    file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file).resize((300, 300), Image.LANCZOS)

elif opt == 'Upload image via link':
    try:
        img = st.text_input('Enter the Image Address')
        image = Image.open(urllib.request.urlopen(img)).resize((300, 300), Image.LANCZOS)
    except:
        if st.button('Submit'):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

try:
    if image is not None:
        st.image(image, width=300, caption='Uploaded Image')
        if st.button('Predict'):
            img = preprocess(image)

            model = model_arc()

            # imported the requests library
            image_url = "https://drive.google.com/file/d/1jc_gp8qR8t6e8z8WOA_K-SqfurN7Z8vn/view?usp=sharing"
            # URL of the image to be downloaded is defined as image_url
            r = requests.get(image_url) # create HTTP response object

            model.load_weights(r)

            prediction = model.predict(img)
            st.info('Hey! The uploaded image has been classified as "{} waste" '.format(labels[np.argmax(prediction)]))
except Exception as e:
    st.info(str(e))

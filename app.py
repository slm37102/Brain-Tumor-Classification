import streamlit as st
import pandas as pd
import urllib.request
from fastai.vision.all import *
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# # for Windows
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

@st.cache
def get_sample():
    df_sample = pd.read_csv('sample_image.csv')
    return df_sample

@st.cache
def get_image(df_sample, image_option):
    df_sample = df_sample[df_sample['BraTS21ID'] == image_option]
    image_path = df_sample['filepath'].values[0]
    actual = df_sample['MGMT_value'].values[0]
    return image_path, actual

# TODO: cache model
# https://docs.streamlit.io/library/advanced-features/caching#typical-hash-functions
@st.cache
def get_model():
    pass

class WrongFileType(ValueError):
    pass

def get_x(r):
    return r['filepath']

def get_y(r):
    return r['MGMT_value']

def dicom2png(file):
    dicom = pydicom.read_file(file, force=True)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    im = Image.fromarray(data)
    return im

image_path = 'Image.png'
learn = load_learner('export.pkl')

header = st.container()
prediction_col, actual_col = st.columns(2)
visualization = st.container()

with header:
    with st.spinner(text="Robot are not train to be slow..."):
        actual = ""
        st.header('Pog Prediction')
        option = st.selectbox(
            'Data Source',
            ('Sample Data', 'Upload Data')
        )
        
        if option == 'Sample Data':
            df_sample = get_sample()
            sample_option = sorted(list(df_sample['BraTS21ID']))
            image_option = st.selectbox(
                'Sample Image ID',
                sample_option
            )
            image_path, actual = get_image(df_sample, image_option)
        
        if option == 'Upload Data':
            dicom_bytes = st.file_uploader("Upload DICOM file")
            if not dicom_bytes:
                raise st.stop()  
            try:
                png = dicom2png(dicom_bytes)
            except:
                st.write(WrongFileType("Does not appear to be a DICOM file"))
                raise st.stop()
            png.save(image_path)

        pred = learn.predict(image_path) # ('1', TensorBase(1), TensorBase([0.0034, 0.9966]))
        prediction = 'No MGMT present' if pred[0] == "0" else "MGMT present"
        actual = 'No MGMT present' if actual == 0 else "MGMT present"
with prediction_col:
    st.metric(
        label="Prediction", 
        value=f"{prediction}", 
        delta=f"Confidence: {round(float(pred[2][int(pred[0])]) * 100, 4)} %"
    )
with actual_col:
    st.metric(
        label="Actual", 
        value=f"{actual}"
    )
with visualization:
    # TODO: make time cost
    st.title('Sus Image')
    st.image(image_path)

st.balloons()
import streamlit as st
from fastai.vision.all import *
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

header = st.container()

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

class WrongFileType(ValueError):
    pass

image_path = 'Image-13.png'
learn = load_learner('export.pkl')

with header:
    st.header('Pog Prediction')
    # option = st.selectbox(
    #     'How would you like to be contacted?',
    #     ('Email', 'Home phone', 'Mobile phone')
    # )
    dicom_bytes = st.file_uploader("Upload DICOM file")

    if not dicom_bytes:
        raise st.stop()  
    try:
        png = dicom2png(dicom_bytes)
    except:
        st.write(WrongFileType("Does not appear to be a DICOM file"))
        raise st.stop()
    st.image(png)
    st.text(learn.predict(image_path))
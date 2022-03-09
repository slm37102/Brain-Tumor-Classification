import streamlit as st
from fastai.vision.all import *
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()

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
    if np.max(data) == 0:
        return None, None, None 
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    if not data.any():
        return None, None, None
        
    im = Image.fromarray(data)
    return im

class WrongFileType(ValueError):
    pass

# @st.cache()
# def load_model(path: str = 'export.pkl'):
#     """Retrieves the trained model and maps it to the CPU by default,
#     can also specify GPU here."""
#     learn = load_learner(path)
#     return learn

path = 'export.pkl'
image_path = 'Image-13.png'
learn = load_learner(path)

with header:
    st.header('Pog Prediction')
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

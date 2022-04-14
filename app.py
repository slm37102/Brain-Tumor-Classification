import streamlit as st
import pandas as pd
from fastai.vision.all import *
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import streamlit.components.v1 as components
import time

start_time = time.time()

# # for Windows
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(initial_sidebar_state='expanded',)

@st.cache(ttl=300)
def get_sample():
    df_sample = pd.read_csv('sample_image.csv')
    return df_sample

# TODO: cache model
# https://docs.streamlit.io/library/advanced-features/caching#typical-hash-functions
# https://docs.streamlit.io/library/advanced-features/experimental-cache-primitives
@st.experimental_singleton
def get_model():
    return load_learner('export.pkl')

class WrongFileType(ValueError):
    pass

def get_x(r):
    return r['filepath']

def get_y(r):
    return r['MGMT_value']

def dicom2png(file):
    """Turn Dicom image into PNG format

    Args:
        file (str): dicom image file that user uploaded 

    Returns:
        Image: Image in PNG format 
    """
    dicom = pydicom.read_file(file, force=True)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    im = Image.fromarray(data)
    return im

def create_animation(ims):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(3,3))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)

learn = get_model()
st.title('🧠 Brain Damaged Estimator 🧠')

# App Description
with st.expander("What is this app for"):
    st.markdown("The dataset is based on a Kaggle competition called [RSNA-MICCAI Brain Tumor Radiogenomic Classification](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification), which is organized by Radiological Society of North America (RSNA).")
    st.write("The app aims to predict the status of a genetic biomarker (MGMT promoter methylation) which is important for choosing the brain cancer treatment for a patient.")
    st.write("MGMT promoter methylation is the key mechanism of MGMT gene silencing and predicts a favorable outcome in patients with glioblastoma who are exposed to alkylating agent chemotherapy.")

with st.expander("How to use"):
    st.write("You can use either sample image or upload a dicom file to predict the result.")
    st.markdown("""Step to use:
1. Open sidebar
2. Select data source
3. Select sample or upload dicom file
4. Press the Predict button""")
    st.write("")
    st.markdown("Sample dicom file can be downloaded from [here](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data).")

# TODO: EDA
with st.expander('Data Visualization'):
    images = [Image.open(f'images/EDA/Image-{i}.png') for i in range(4,33)]
    line_ani = create_animation(images)
    components.html(line_ani.to_jshtml().replace('''.anim-state label {
    margin-right: 8px;
}''', '''.anim-state label {
    margin-right: 8px;
    color: white;
    font-family:Helvetica;
    font-size: 12px;
}'''), height=400)

header = st.container()
prediction_col, actual_col = st.columns(2)
visualization = st.container()

with st.sidebar:
    st.header("Data Selection")
    with st.spinner(text="Robot are not train to be slow..."):
        actual = ""
        option = st.radio(
            'Select Your Data Source',
            ('Sample Data', 'Upload Data')
        )

        # if user chooses to use sample data
        if option == 'Sample Data':
            df_sample = get_sample()
            sample_option = sorted(list(df_sample['BraTS21ID']))
            with st.expander("List of sample data"):
                # loop through all sample images
                for _, row in df_sample.iterrows():
                    st.image(row['filepath'], caption=f"Sample Image ID: {row['BraTS21ID']}, MGMT value: {row['MGMT_value']}")
            image_option = st.selectbox(
                'Sample Image ID',
                sample_option
            )
            image_path = df_sample[df_sample['BraTS21ID'] == image_option]['filepath'].values[0]
            actual = df_sample['MGMT_value'].values[0]
        
        # if user chooses to use upload data
        if option == 'Upload Data':
            image_path = 'image.png'
            dicom_bytes = st.file_uploader("Upload DICOM file")
            # if the upload file is empty
            if not dicom_bytes:
                raise st.stop()  
            # try if file format is dicom 
            try:
                png = dicom2png(dicom_bytes)
            except:
                st.write(WrongFileType("Does not appear to be a DICOM file"))
                raise st.stop()
            
            png.save(image_path)
        
        pressed = st.button('Do not click')

# if button is pressed
if pressed:
    with st.spinner(text="Predicting in Progress..."):
        # predict using image
        pred = learn.predict(image_path) # sample output = ('1', TensorBase(1), TensorBase([0.0034, 0.9966]))
        prediction = 'No MGMT present' if pred[0] == "0" else "MGMT present"
        actual = 'No MGMT present' if actual == 0 else "MGMT present"
        
        with header:
            st.header("Pog Prediction")

        with visualization:
            st.write("Time taken: %.3f seconds" % (time.time() - start_time))
            st.image(image_path)
            
        with prediction_col:
            st.metric(
                label="Predicted", 
                value=f"{prediction}", 
                delta=f"Confidence: {round(float(pred[2][int(pred[0])]) * 100, 4)} %"
            )
            
        with actual_col:
            st.metric(
                label="Actual", 
                value=f"{actual}"
            )

        st.balloons()
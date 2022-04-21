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
import s3fs
import os

# # for Windows
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# initiate AWS S3 filesystem
fs = s3fs.S3FileSystem(anon=False)
# show sidebar when entering the app
st.set_page_config(
    page_title = 'Brain Tumour Radiogenomic Classification', 
    page_icon = '🧠',
    initial_sidebar_state = 'expanded')

class WrongFileType(ValueError):
    pass

@st.cache(ttl=600)
def get_sample(filename='sample_image.csv'):
    """Get the sample image dataset from csv file
    The function is cached for 5 minutes for faster runtime.

    Args:
        filename (str, optional): the file name to read. Defaults to 'sample_image.csv'.

    Returns:
        pandas.DataFrame: data frame of sample images
    """
    df_sample = pd.read_csv(filename)
    return df_sample

@st.experimental_singleton
def get_model(filename='export.pkl'):
    """Download the model from AWS S3. return learner for prediction.

    Args:
        filename (str, optional): filename of the model to be imported. Defaults to 'export.pkl'.

    Returns:
        Learner: Fastai Learner class
    """
    filename = 'models/' + filename
    # download model from the AWS S3 bucket filesystem
    fs.get(f'fyp-slm/{filename}', filename)
    # load model
    model = load_learner(filename)
    # remove model file
    os.remove(filename)
    return model

# neccesary input function for importing the model
def get_x(r):
    return r['filepath']

# neccesary output function for importing the model
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

@st.cache(ttl=600)
def get_images():
    """Get images of sample data

    Returns:
        list: list of Image
    """
    return [Image.open(f'images/EDA/Image-{i}.png') for i in range(4,33)]

def create_animation():
    """create animation using the brain tumour images

    Returns:
        animation.FuncAnimation: animation.FuncAnimation from matplotlib
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(3,3))
    plt.axis('off')
    images = get_images()
    im = plt.imshow(images[0], cmap="gray")

    def animate_func(i):
        im.set_array(images[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames = len(images), interval = 1000//24)

st.title('Brain Tumour Radiogenomic Classification')

# App Description
with st.expander("About the app"):
    st.markdown("This app is based on a Kaggle competition called [RSNA-MICCAI Brain Tumor Radiogenomic Classification](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification), which is organized by Radiological Society of North America (RSNA).")
    st.markdown("The app aims to predict the status of a genetic biomarker, ***MGMT promoter methylation***,  which is important for choosing the brain cancer treatment for a patient.")
    st.markdown("***MGMT promoter methylation*** is the key mechanism of MGMT gene silencing and predicts a favorable outcome in patients with glioblastoma who are exposed to alkylating agent chemotherapy.")

# App Manual
with st.expander("How to use"):
    st.write("You can use either sample image or upload a dicom file to predict the result.")
    st.markdown("""> Step to use:
> 1. Open sidebar
> 2. Select data source
> 3. Select sample / upload a dicom file (can be downloaded from [here](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data))
> 4. Press the \"Start Predict\" button to start the prediction""")

# content of sidebar
with st.expander('Data Visualization'):
    st.write('Sample Magnetic Resonance Imaging (MRI) image of brain tumour')
    # loading spinner to show process in progress
    with st.spinner(text="Robot are not train to be slow..."):
        # create animation for data visualization
        line_ani = create_animation()
        # make component html to show animation
        components.html(line_ani.to_jshtml().replace('''.anim-state label {
    margin-right: 8px;
}''', '''.anim-state label {
    margin-right: 8px;
    color: white;
    font-family:Helvetica;
    font-size: 12px;
}'''), height=400)

with st.expander("More about the project"):
    st.markdown("You can go to the [GitHub link](https://github.com/slm37102/Brain-Tumor-Classification) to learn more about the project.")
    st.write("The model training is done using [this Kaggle notebook](https://www.kaggle.com/code/slm37102/t1w-brain-tumor-eca-nfnet-l2-5-epoch)")
    st.write("The model is trained based on [this competition dataset](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)")

# set up layout of the app
header = st.container()
prediction_col, actual_col = st.columns(2)
visualization = st.container()

with st.sidebar:
    st.header("Data Selection")
    # loading spinner to show process in progress
    with st.spinner(text="Robot are not train to be slow..."):
        # if user upload dicom file, actual is empty
        actual = ""
        # radio buttion for user to select the test data source
        option = st.radio(
            'Select Your Data Source',
            ('Sample Data', 'Upload Data'))

        # if user chooses to use sample data
        if option == 'Sample Data':
            # get list of sample images
            df_sample = get_sample()
            sample_option = sorted(list(df_sample['BraTS21ID']))
            
            with st.expander("List of sample data"):
                # loop through all sample images and show images
                for _, row in df_sample.iterrows():
                    st.image(
                        row['filepath'], 
                        caption=f"Image ID: {row['BraTS21ID']}, MGMT value: {'MGMT not present' if row['MGMT_value'] == 0 else 'MGMT present'}")
            # drop down box option for sample images
            image_option = st.selectbox(
                'Sample Image ID',
                sample_option,
                help='Select a sample image to predict, the sample image can be seen in the list above'
            )
            # filter to selected_df
            selected_df = df_sample[df_sample['BraTS21ID'] == image_option]
            # get path of the sample image selected
            image_path = selected_df['filepath'].values[0]
            # get actual value of the selected sample image
            actual = selected_df['MGMT_value'].values[0]
        
        # if user chooses to use upload data
        if option == 'Upload Data':
            image_path = 'image.png'
            dicom_bytes = st.file_uploader("Upload DICOM file")
            # if the upload file is empty
            if not dicom_bytes:
                raise st.stop()  
            # try if file format is dicom 
            try:
                # convert dicom file to png format
                png = dicom2png(dicom_bytes)
            # if file format is not dicom, show error
            except:
                st.write(WrongFileType("Does not appear to be a DICOM file"))
                raise st.stop()
            # save uploaded image to path
            png.save(image_path)

        # checkbox to let user choose if needs to show image
        display_image = st.checkbox(
            'Display image',
            help='Display the upload/selected image')
        # checkbox to let user choose if needs to show time taken
        display_time = st.checkbox(
            'Display time taken', 
            help='Display time needed to perform the prediction')
        # button to start predict
        pressed = st.button('Start Predict')

# if 'Start Predict' button is pressed
if pressed:
    # loading spinner to show process in progress
    with st.spinner(text="Downloading Model..."):
        # get model for prediction
        learn = get_model()
    
    # initialize start_time
    start_time = time.time()
    
    # loading spinner to show process in progress
    with st.spinner(text="Predicting in Progress..."):
        # predict using image
        pred = learn.predict(image_path) # sample output = ('1', TensorBase(1), TensorBase([0.0034, 0.9966]))
        # prediction output to label value
        prediction = 'MGMT not present' if pred[0] == "0" else "MGMT present"
        # actual output to label value
        actual = 'MGMT not present' if actual == 0 else "MGMT present"
        
        with header:
            st.header("Prediction result")

        with visualization:
            # displaytime taken for prediction
            if display_time: 
                st.success("Time taken to predict: %.3f seconds" % (time.time() - start_time))
            # display image for prediction
            if display_image:    
                st.header("Image")
                st.image(image_path)
        
        with prediction_col:
            # show predicted output and confidence
            st.metric(
                label="Predicted", 
                value=f"{prediction}", 
                delta=f"Confidence: {round(float(pred[2][int(pred[0])]) * 100, 4)} %"
            )
            
        if option == 'Sample Data':
            with actual_col:
                # show actual output 
                st.metric(
                    label="Actual", 
                    value=f"{actual}"
                )

        # show balloons when finished
        st.balloons()
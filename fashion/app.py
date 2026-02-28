import pandas as pd
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st


# -------------------------
# basic config
# -------------------------
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.header("Fashion Recommendation System")


# -------------------------
# base directory (VERY IMPORTANT for Streamlit Cloud)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------
# create upload folder safely
# -------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# -------------------------
# load model (cached)
# -------------------------
@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPool2D()
    ])

    return model


# -------------------------
# load features safely (cached)
# -------------------------
@st.cache_resource
def load_features():

    features_path = os.path.join(BASE_DIR, "Images_features.pkl")
    filenames_path = os.path.join(BASE_DIR, "filenames.pkl")

    Image_features = pkl.load(open(features_path, "rb"))
    filenames = pkl.load(open(filenames_path, "rb"))

    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(Image_features)

    return Image_features, filenames, neighbors


model = load_model()
Image_features, filenames, neighbors = load_features()


# -------------------------
# feature extraction
# -------------------------
def extract_features(img_path, model):

    img = image.load_img(img_path, target_size=(224,224))

    img_array = image.img_to_array(img)

    expanded = np.expand_dims(img_array, axis=0)

    preprocessed = preprocess_input(expanded)

    result = model.predict(preprocessed, verbose=0).flatten()

    normalized = result / norm(result)

    return normalized


# -------------------------
# upload UI
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    st.subheader("Uploaded Image")
    st.image(uploaded_file, width=250)


    features = extract_features(file_path, model)

    distances, indices = neighbors.kneighbors([features])


    st.subheader("Recommended Images")

    cols = st.columns(5)

    for i in range(5):

        img_path = os.path.join(BASE_DIR, filenames[indices[0][i+1]])

        cols[i].image(img_path)
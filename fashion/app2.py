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

st.header('Fashion Recommendation System')

Image_features=pkl.load(open('Images_features.pkl','rb'))
filenames=pkl.load(open('filenames.pkl','rb'))

# st.write(type(filenames))
# st.write(len(filenames) if filenames is not None else "filenames is None")

# st.write(f"Type: {type(filenames)}")
# st.write(f"Total files: {len(filenames)}")

def extract_features_from_images(image_path,model):
    img=image.load_img(image_path, target_size=(224,224))
    img_array=image.img_to_array(img)
    img_expand_dim=np.expand_dims(img_array,axis=0)
    # img_expand_dim.shape
    img_preprocess=preprocess_input(img_expand_dim)
    # img_preprocess.shape
    result=model.predict(img_preprocess).flatten()
    # result.shape
    norm_result=result/norm(result)
    return norm_result

model=ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable=False

model=tf.keras.models.Sequential([model,GlobalMaxPool2D()])

neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(Image_features)

upload_file=st.file_uploader("Upload Image")

if upload_file is not None:
    file_path=os.path.join('upload',upload_file.name)
    with open(file_path,'wb') as f:
        f.write(upload_file.getbuffer())
    st.subheader('Uploaded Image')
    st.image(upload_file)
    input_img_features=extract_features_from_images(file_path,model)
    distances, indices=neighbors.kneighbors([input_img_features])
    st.subheader('Recommended Images')
    c1,c2,c3,c4,c5=st.columns(5)
    with c1:
        st.image(filenames[indices[0][1]])
    with c2:
            st.image(filenames[indices[0][2]])
    with c3:
            st.image(filenames[indices[0][3]])
    with c4:
            st.image(filenames[indices[0][4]])
    with c5:
            st.image(filenames[indices[0][5]])

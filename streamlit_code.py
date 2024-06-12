import streamlit as st 
import os 
from PIL import Image
import pickle
import tensorflow 
import numpy as np
from numpy.linalg import norm 
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2 
import pandas as pd 
from keras.models import Sequential

feature_list = np.array(pickle.load(open('features_encoding.pkl','rb')))
filenames = pickle.load(open('product.pkl','rb'))  
try:
    df = pd.read_csv(
        r'C:\Users\JAY\Desktop\E-Commerce_Fashion_Recommendation\myntradataset\styles.csv',
        encoding='utf-8',     # Try 'utf-8' encoding
        delimiter=',',        # Specify delimiter if necessary
        on_bad_lines='skip',  # Skip bad lines
                   # Read only the first 10 rows
    )
    print(df.head())
except Exception as e:
    print(f"Error reading the CSV file: {e}")


model = Sequential()
pre_trained = ResNet50(include_top = False, weights='imagenet',input_shape=(224,224,3))
for layer in pre_trained.layers:
  layer.trainable = False
model.add(pre_trained)
model.add(GlobalMaxPooling2D())

df = pd.read_csv('styles.csv', on_bad_lines='skip')

def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224,3)) 
    img = np.array(img)
    exp_img = np.expand_dims(img,axis =0)
    preprocess_img=preprocess_input(exp_img)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result) 
    

    return normalized_result 
def recommended_images(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list) 

    distances,indices = neighbors.kneighbors([features]) 
    recommeded = [] 
    matter = [] 
   
    for id in indices[0]:
        Img_path = os.path.join(r'C:\Users\JAY\Desktop\E-Commerce_Fashion_Recommendation\static\myntradataset\images',filenames[id])  
        a = df[df['id'] == id]['productDisplayName'].to_string(index=False)
        matter.append(a)
        recommeded.append(Img_path)  
    

    return recommeded,matter



st.title('Fashion Recommeder System')  
upload_File = st.file_uploader('Upload Your Image') 
def Save_upload(upload_File):
    try:
        with open(os.path.join( 'Uploads',upload_File.name),'wb') as f:
            f.write(upload_File.getbuffer()) 
        return 1 
    except:
        return 0
    
if upload_File is not None:
    if Save_upload(upload_File): 
        display_Image = Image.open(upload_File) 
        st.image(display_Image) 
        features =feature_extraction(os.path.join('Uploads',upload_File.name),model)
        #st.text(features)

        #recommendations 
        recommended_images_list,matter = recommended_images(features,feature_list)    
        
        st.text('Recommended Product')
        #st.text(recommended_images_list)
        col1,col2,col3,col4,col5  = st.columns(5) 
        with col1:
            st.image(Image.open(recommended_images_list[1]))
            description = matter[1] 
            st.text(description)
        with col2:
            st.image(recommended_images_list[2]) 
            description = matter[2] 
            st.text(description)

        with col3:
            st.image(recommended_images_list[3]) 
            description = matter[3] 
            st.text(description)
        with col4:
            st.image(recommended_images_list[4]) 
            description = matter[4] 
            st.text(description)
        with col5:
            st.image(recommended_images_list[5])
            description = matter[5  ] 
            st.text(description)
    else:
        st.header('Some Error in Uploading File')
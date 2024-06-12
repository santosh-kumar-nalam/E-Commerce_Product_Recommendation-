import pickle
import tensorflow 
import os
import numpy as np
from numpy.linalg import norm 
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
from keras.models import Sequential


feature_list = np.array(pickle.load(open('features_encoding.pkl','rb')))
filenames = pickle.load(open('product.pkl','rb'))

model = Sequential()
pre_trained = ResNet50(include_top = False, weights='imagenet',input_shape=(224,224,3))
for layer in pre_trained.layers:
  layer.trainable = False
model.add(pre_trained)
model.add(GlobalMaxPooling2D())


img_path = r'C:\Users\JAY\Desktop\E-Commerce_Fashion_Recommendation\sample_images\watch2.jpg'
img = image.load_img(img_path,target_size =(224,224))
img = np.array(img)
exp_img = np.expand_dims(img,axis =0)
preprocess_img=preprocess_input(exp_img)
result = model.predict(preprocess_img).flatten()
normalized_result = result / norm(result)



neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list) 


distances,indices = neighbors.kneighbors([normalized_result])
print(indices)  

recommeded = []
for id in indices[0]:
    Img_path = os.path.join(r'C:\Users\JAY\Desktop\E-Commerce_Fashion_Recommendation\myntradataset\images',filenames[id])  
    recommeded.append(Img_path)  
print(recommeded) 

for file in recommeded:
    temp_img = cv2.imread(file)
    cv2.imshow('output',cv2.resize(temp_img,(224,224)))
    cv2.waitKey(0)
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from numpy.linalg import norm
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import pickle
import pandas as pd

# Load data
df = pd.read_csv('styles.csv', on_bad_lines='skip')
feature_list = np.array(pickle.load(open('features_encoding.pkl', 'rb')))
filenames = pickle.load(open('product.pkl', 'rb'))

# Initialize model
model = Sequential()
pre_trained = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in pre_trained.layers:
    layer.trainable = False
model.add(pre_trained)
model.add(GlobalMaxPooling2D())

# Function to extract features
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224, 3))
    img = np.array(img)
    exp_img = np.expand_dims(img, axis=0)
    preprocess_img = preprocess_input(exp_img)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to find recommended images
def recommended_images(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    recommended = []
    matter = []
    for idx in indices[0]:  
        img_path = os.path.join('static', 'myntradataset', 'images', filenames[idx])
        display_name = df[df['id'] == int(filenames[idx].split('.')[0])]['productDisplayName'].values
        if len(display_name) > 0:
            display_name = display_name[0]
        else:
            display_name = "Unknown"
        matter.append(display_name)
        recommended.append(filenames[idx])
    return matter[1:], recommended[1:]

def similar(uploaded_file):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    features = feature_extraction(img_path, model)
    matter, recommended_images_list = recommended_images(features, feature_list)
    return recommended_images_list, matter


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            rec_images, matter = similar(file.filename)   
            
            return render_template('main.html', rec_images=rec_images) 
    return render_template('main.html')  

if __name__ == '__main__':
    app.run(debug=True)

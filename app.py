import os
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def compare_images(original_image, new_image, model):
    original_features = model.predict(original_image)
    new_features = model.predict(new_image)
    similarity_score = cosine_similarity(original_features, new_features)[0][0]
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # Check if two files were uploaded
    if 'file1' not in request.files or 'file2' not in request.files:
        return "Please upload both files"

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Checking if the filenames are empty
    if file1.filename == '' or file2.filename == '':
        return "Please select both files"

    # Check if the file extensions are allowed
    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return "File extensions not allowed"

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    file1.save(file1_path)
    file2.save(file2_path)

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    original_image = preprocess_image(file1_path)
    new_image = preprocess_image(file2_path)

    # Compare the images
    similarity_score = compare_images(original_image, new_image, model)


    os.remove(file1_path)
    os.remove(file2_path)

    return "Similarity Score: {:.4f}".format(similarity_score)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)

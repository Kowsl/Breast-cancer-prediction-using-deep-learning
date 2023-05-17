import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from PIL import Image, ImageEnhance,ImageDraw,ImageFont,ImageOps
from tensorflow.keras.models import load_model

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('breastcancer.h5')
IMAGE_SIZE=128
unique_labels = ['benign','malignant']


def predict_image(image_path, model):
    # Load the image
    img = Image.open(image_path)
    img=img.convert('RGB')
    # Preprocess the image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    # Make predictions
    pred = model.predict(np.array([img]))
    label = unique_labels[np.argmax(pred)]
    return label

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST','GET'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    text1='Result: '
    file = request.files['file']

    # Save the file to the server
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    # Make predictions
    label = predict_image(file, model)
    text2="The Pathology report for Breast shows", label

    return render_template('home.html', filename=file.filename, label=label,occurance=text1,child_occur=text2,image=file)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

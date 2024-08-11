import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('bestmodel.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = detect_pneumonia(file_path)
            return jsonify({'result': result, 'image_path': url_for('static', filename=filename)})
        return jsonify({'error': 'File not allowed'})
    return render_template('index.html')

def detect_pneumonia(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)

    if class_idx == 0:
        return "PNEUMONIA"
    else:
        return "NORMAL"

if __name__ == '__main__':
    app.run(debug=True)

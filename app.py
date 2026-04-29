from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app) # Enable CORS for all routes

MODEL_PATH = 'models/mobilenetv2.h5'
IMG_SIZE = (224, 224)

try:
    print("Loading AI model... please wait.")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_class_names():
    data_dir = 'data'
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        return classes
    return ['Bacterial leaf blight', 'Brown spot', 'Leaf smut'] 

CLASS_NAMES = get_class_names()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:

        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

       
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(image_rgb, IMG_SIZE)
        img_input = img_input / 255.0
        img_input = np.expand_dims(img_input, axis=0)


        predictions = model.predict(img_input)
        predicted_idx = np.argmax(predictions[0])
        disease = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])

        return jsonify({
            "disease": disease,
            "confidence": confidence
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask server
    print("\n--- Rice Disease Prediction Server ---")
    print("Go to http://127.0.0.1:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

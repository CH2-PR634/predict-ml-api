import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from rembg import remove
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

model = load_model('model.h5')


def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")


def remove_background(input_path, output_path):
    try:
        input_image = cv2.imread(input_path)
        output_image = remove(input_image)
        cv2.imwrite(output_path, output_image)
    except Exception as e:
        raise ValueError(f"Error during background removal: {str(e)}")


@app.route('/')
def index():
    return jsonify({'status': 'success', 'message': 'OK'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        temp_path = 'temp_image.jpg'
        file.save(temp_path)

        preprocess_image(temp_path)
        remove_background(temp_path, 'output_image.jpg')

        predictions = model.predict(preprocess_image(temp_path))
        predicted_class = np.argmax(predictions)
        max_predict_points = predictions.max()

        alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        result = {
            'predicted_class': int(predicted_class),
            'huruf': alphabet[int(predicted_class)],
            'predict_points': str(max_predict_points),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = "C:/Users/shyam/Downloads/MNIST_model.h5"
model = load_model(model_path)

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST standard size
    image = np.array(image)
    image = image.reshape((1, 28, 28, 1)).astype('float32') / 255.0  # Normalize
    return image

index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Digit Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 500px;
            margin: 100px auto;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            padding: 40px;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        #file-input {
            display: none;
        }
        label {
            display: block;
            padding: 20px;
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        label:hover {
            background-color: #0056b3;
        }
        #result-box {
            margin-top: 20px;
            padding: 20px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }
        #result-box p {
            font-size: 20px;
            color: #333;
            margin: 0;
        }
        #result-box strong {
            color: #007bff;
            font-weight: bold;
        }
        #submit-btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        #submit-btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Handwritten Digit Recognition</h1>
        <form id="upload-form">
            <label for="file-input">Choose Image</label>
            <input type="file" id="file-input" accept=".png, .jpg, .jpeg" required>
            <h5> </h5>
            <button type="submit" id="submit-btn">Submit</button>
        </form>
        <div id="result-box"></div>
    </div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', function(event) {
            event.preventDefault();
            var fileInput = document.getElementById('file-input');
            var file = fileInput.files[0];
            var formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result-box').innerHTML = '<p>The predicted digit is: <strong>' + data.prediction + '</strong></p>';
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return index_html

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream)
            processed_img = preprocess_image(img)
            prediction = model.predict(processed_img).argmax()
            return jsonify({'prediction': str(prediction)})
    return jsonify({'prediction': 'Error'})

if __name__ == '__main__':
    app.run(debug=True)

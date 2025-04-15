from flask import Flask, render_template_string, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image 

# Initialize the Flask application
app = Flask(__name__)

# Path to model (replace with your correct model file path)
model_path ='C:\\Users\\aayus\\Documents\\Brahmastra\\Jackpot\\Flask\\hu\\model.pkl'
model = load_model(model_path)

# Set up allowed file types and upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'C:\\Users\\aayus\\Documents\\Brahmastra\\Jackpot\\Flask\\hu\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to home page
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Preprocess the image for prediction
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Assuming the model expects 224x224 input
        img_array = np.array(img) / 255.0  # Normalizing if required
        img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        result = np.argmax(prediction, axis=1)  # Adjust based on your model's output

        # Return the result (you can customize this to display more info)
        return render_template_string(RESULT_TEMPLATE, result=result[0])

    return "File not allowed"

if __name__ == '__main__':
    app.run(debug=True)


# HTML and CSS Combined Template for Home Page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
        }
        .container {
            width: 50%;
            margin: 50px auto;
            text-align: center;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload an Image for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br>
            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>
'''

# HTML and CSS Combined Template for Result Page
RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
        }
        .container {
            width: 50%;
            margin: 50px auto;
            text-align: center;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .result {
            font-size: 22px;
            color: #333;
            margin-top: 20px;
        }
        a {
            text-decoration: none;
            color: #4CAF50;
            font-size: 18px;
            margin-top: 20px;
            display: inline-block;
        }
        a:hover {
            color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Result</h1>
        <div class="result">
            <p>Your image was predicted to be: <strong>{{ result }}</strong></p>
        </div>
        <a href="/">Upload another image</a>
    </div>
</body>
</html>
'''


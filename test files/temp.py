from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.models import load_model

# Load the trained InceptionV3 model
model = load_model("InceptionV3.h5")

# Define class labels
class_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus', 'Ringworm']

# Render index.html page
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input image from client, then predict class and render respective .html page for the solution
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print("@@ Input posted =", filename)

        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class...")
        pred = predict_class(file_path)

        # Find the corresponding HTML file based on the predicted class
        output_page = pred.lower() + '.html'

        return render_template(output_page, pred_output=pred, user_image=file_path)

def predict_class(image_path):
    test_image = load_img(image_path, target_size=(224, 224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = inceptionv3_preprocess_input(test_image)
    
    prediction = model.predict(test_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    return predicted_class

# For local system & cloud
if __name__ == "__main__":
    app.run(debug=True)

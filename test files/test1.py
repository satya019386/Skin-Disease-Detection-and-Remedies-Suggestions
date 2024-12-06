from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.models import load_model
import cv2

# Load the trained InceptionV3 and MobileNet models
inception_model = load_model("model/InceptionV3.h5")
mobilenet_model = load_model("model/MobileNet.h5")

# Define class labels
class_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus', 'Ringworm']

# Render index.html page
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_skin(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image file")

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for the skin color in HSV
    lower_bound = np.array([0, 48, 80], dtype=np.uint8)
    upper_bound = np.array([20, 255, 255], dtype=np.uint8)
    # Create a mask to identify skin regions
    skin_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    # Count the number of white pixels (skin) in the mask
    skin_pixels = cv2.countNonZero(skin_mask)
    # Calculate the percentage of skin pixels in the image
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    skin_ratio = skin_pixels / total_pixels
    # Return True if the skin ratio is above a threshold, indicating that the image contains skin
    return skin_ratio > 0.15

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

def preprocess_image(image_path):
    test_image = load_img(image_path, target_size=(224, 224))
    test_image = img_to_array(test_image)
    return test_image

def predict_class(model, test_image):
    test_image = np.expand_dims(test_image, axis=0)
    if model == inception_model:
        test_image = inceptionv3_preprocess_input(test_image.copy())
    elif model == mobilenet_model:
        test_image = mobilenet_preprocess_input(test_image.copy())
    else:
        raise ValueError("Invalid model type. Supported models are 'InceptionV3' and 'MobileNet'.")
    
    prediction = model.predict(test_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    return predicted_class

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if not allowed_file(file.filename):
            return render_template('index.html', error='Only images with extensions png, jpg, jpeg, or gif are allowed')

        filename = file.filename
        print("@@ Input posted =", filename)

        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)

        print("@@ Detecting skin...")
        try:
            if not detect_skin(file_path):
                return render_template('index.html', error='The uploaded image does not contain skin regions.')
            
            print("@@ Preprocessing image...")
            test_image = preprocess_image(file_path)

            print("@@ Predicting class with InceptionV3...")
            inception_pred = predict_class(inception_model, test_image)

            print("@@ Predicting class with MobileNet...")
            mobilenet_pred = predict_class(mobilenet_model, test_image)

            # Render the corresponding HTML template based on the predicted classes
            inception_output = inception_pred.lower() + '.html'
            mobilenet_output = mobilenet_pred.lower() + '.html'

            return render_template(inception_output, inception_pred=inception_pred, mobilenet_pred=mobilenet_pred, user_image=file_path)
        
        except Exception as e:
            return render_template('index.html', error='An error occurred during prediction: {}'.format(str(e)))

# For local system & cloud
if __name__ == "__main__":
    app.run(debug=True)

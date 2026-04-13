from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("models/model.h5")

# Class names (must match your training folders)
classes = ["covid", "normal", "pneumonia"]

# Folder to save uploads
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------
# Prediction function
# --------------------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.reshape(img, (1,150,150,3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return classes[class_index]

# --------------------------
# Home Page
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --------------------------
# Predict Route
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    result = predict_image(path)

    return render_template('index.html', prediction=result)

# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
    import os

port = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
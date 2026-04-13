import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/model.h5")

# Class labels (must match folder names EXACTLY)
classes = ["covid", "normal", "pneumonia"]

def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Image not found. Check path!")
        return None

    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.reshape(img, (1,150,150,3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return classes[class_index]

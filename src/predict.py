import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/model.h5")

# Classes (must match folder names)
classes = ["covid", "normal", "pneumonia"]

def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return None

    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.reshape(img, (1,150,150,3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return classes[class_index]

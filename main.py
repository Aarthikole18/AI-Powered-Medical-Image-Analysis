import cv2
import matplotlib.pyplot as plt
from src.predict import predict_image

path = r"data/train/normal/00000553_000.png"

# Predict
result = predict_image(path)

# Show image
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title(f"Prediction: {result}")
plt.axis("off")
plt.show()
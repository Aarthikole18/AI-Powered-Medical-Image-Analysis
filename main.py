import cv2
import matplotlib.pyplot as plt
from src.predict import predict_image

# ======================
# IMAGE PATH (CHANGE THIS)
# ======================
image_path = r"data/train/normal/00000553_000.png"

# Predict
result = predict_image(image_path)

# Load image for display
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show image + prediction
plt.imshow(img)
plt.title(f"Prediction: {result}")
plt.axis("off")
plt.show()

print("Prediction:", result)
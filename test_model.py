import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("emotion_model2.h5")

# Load an image for testing
img_path = "C:/Users/common-research/Desktop/test2.jfif"  # Change to a real image path
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)  # Add channel dimension
img = img / 255.0  # Normalize

# Predict
prediction = model.predict(img)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
print("Predicted Emotion:", emotion_classes[np.argmax(prediction)])

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os
import time

# Load model once
model = tf.keras.models.load_model("signature_detector_v2.h5")

def clean_signature(img_path, output_dir="static/output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Unique filename based on timestamp
    filename = f"digital_signature_{int(time.time())}.png"
    output_path = os.path.join(output_dir, filename)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    rgba = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGBA)
    rgba[:, :, 3] = clean  # Alpha channel

    cv2.imwrite(output_path, rgba)
    return filename

def predict_signature(img_path, threshold=0.6):
    # Preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    is_sig = pred > threshold
    confidence = pred * 100 if is_sig else (1 - pred) * 100

    return {
        "is_signature": is_sig,
        "confidence": round(confidence, 2),
        "raw_output": round(pred, 4),
        "threshold": threshold
    }

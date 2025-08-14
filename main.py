import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os

model = tf.keras.models.load_model("signature_detector_v2.h5")

OUTPUT_DIR = "digital_signatures_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_next_filename():
    counter = 1
    while True:
        filename = f"digital_signature_{counter}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1

def clean_signature(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    rgba = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGBA)
    rgba[:, :, 3] = clean

    output_path = get_next_filename()
    cv2.imwrite(output_path, rgba)
    print(f"Digital signature saved as {output_path}")

def predict_and_clean(img_path, threshold=0.5):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]

    if pred > threshold:
        print("Detected: Signature")
        print(f"Confidence: {pred*100:.2f}%")
        clean_signature(img_path)
    else:
        print("Detected: Not a Signature")
        print(f"Confidence: {(1-pred)*100:.2f}%")

predict_and_clean("EG_1.jpg", threshold=0.6)
predict_and_clean("EG_2.jpg", threshold=0.6)
predict_and_clean("EG_3.png", threshold=0.6)

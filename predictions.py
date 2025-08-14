import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from termcolor import colored

# Load the trained model
model = tf.keras.models.load_model("signature_detector_v2.h5")

def is_signature(img_path, threshold=0.5, show_img=True):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)[0][0]
    confidence = pred if pred > 0.5 else 1 - pred

    # Determine label
    if pred > threshold:
        label = "Signature"
        color = "green"
    else:
        label = "Not Signature"
        color = "red"

    # Print detailed result
    print(colored(f"Prediction: {label}", color, attrs=["bold"]))
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Model Output: {pred:.4f}")
    print(f"Decision Threshold: {threshold}")

    # Show image with prediction
    if show_img:
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{label} ({confidence*100:.1f}%)", color=color)
        plt.show()

    return label, confidence

# Example usage
is_signature("EG_1.jpg", threshold=0.6)

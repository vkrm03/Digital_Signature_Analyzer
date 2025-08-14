import os
import uuid
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, send_from_directory

# Load Model
model = tf.keras.models.load_model("signature_detector_v2.h5")

# Create uploads and outputs folders if not exist
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/digitalized"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def clean_signature(img_path, output_filename):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    rgba = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGBA)
    rgba[:, :, 3] = clean

    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, rgba)
    return output_filename

def predict_image(img_path, threshold=0.6):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]

    result = {
        "prediction": "Signature" if pred > threshold else "Not Signature",
        "confidence": pred if pred > threshold else 1 - pred,
        "raw_output": pred,
        "threshold": threshold
    }
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            result = predict_image(filepath)

            digitalized_filename = None
            if result["prediction"] == "Signature":
                digitalized_filename = clean_signature(filepath, "digital_" + filename)

            return render_template("index.html", result=result,
                                   uploaded_image=filename,
                                   digitalized_image=digitalized_filename)

    return render_template("index.html", result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/digitalized/<filename>')
def digitalized_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)

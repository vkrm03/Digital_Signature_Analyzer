import cv2
import numpy as np

def clean_signature(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    rgba = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGBA)
    rgba[:, :, 3] = clean  # use binary mask as alpha

    cv2.imwrite("digital_signature.png", rgba)
    print("✅ Signature saved as digital_signature.png")

clean_signature("img1.jpg")

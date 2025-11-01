"""
preprocess.py
Handles image conversion, denoising, and contrast enhancement.
"""

import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement using CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)

    return gray, enhanced

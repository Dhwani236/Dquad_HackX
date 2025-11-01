"""
analyze.py
Performs pixel-level analysis to identify cracks and dents
based on grayscale intensity variation and edge features.
"""

import cv2
import numpy as np
from skimage import filters, morphology, feature

def detect_defects(image):
    # Step 1: Adaptive thresholding (to highlight darker dents)
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # Step 2: Edge detection (to capture microcracks)
    edges = cv2.Canny(image, threshold1=40, threshold2=120)

    # Step 3: Combine threshold + edges
    combined = cv2.bitwise_or(thresh, edges)

    # Step 4: Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 5: Contour detection
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cleaned, contours

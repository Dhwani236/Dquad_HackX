"""
visualize.py
-------------
Displays and saves the final image with detected regions highlighted.
"""

import cv2
import numpy as np

def visualize_results(original, contours, save_path):
    output = original.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite(save_path, output)
    return output

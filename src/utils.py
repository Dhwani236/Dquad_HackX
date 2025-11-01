"""
utils.py
Contains utility functions for debugging and visualization.
"""

import cv2
import matplotlib.pyplot as plt

def show_histogram(image):
    plt.hist(image.ravel(), 256, [0,256])
    plt.title("Grayscale Intensity Distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.show()

def resize_image(image, width=800):
    h, w = image.shape[:2]
    ratio = width / w
    return cv2.resize(image, (width, int(h * ratio)))

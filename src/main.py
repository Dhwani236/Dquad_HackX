"""
main.py
---------
Entry point for grayscale analysis of silicon wafer surface.
Detects microcracks and dents using OpenCV and scikit-image.
"""

import cv2
import numpy as np
from preprocess import preprocess_image
from analyze import detect_defects
from visualize import visualize_results

# Configuration

IMAGE_PATH = "../data/sample_wafer.jpg"  # change path as needed
OUTPUT_PATH = "../data/output_detected.jpg"

def main():
    print("🔍 Loading wafer image...")
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("❌ Error: Image not found. Check your path.")
        return

    print("🧪 Preprocessing image...")
    gray, filtered = preprocess_image(image)

    print("📈 Detecting defects...")
    defect_mask, contours = detect_defects(filtered)

    print("🖼️ Visualizing results...")
    output = visualize_results(image, contours, OUTPUT_PATH)

    print(f"✅ Defect detection complete. Saved output at: {OUTPUT_PATH}")
    cv2.imshow("Detected Defects", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

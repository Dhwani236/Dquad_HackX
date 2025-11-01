# TeamName — Dquad

## 🪐 Problem Statement

In the pre-fabrication stage of microchip manufacturing that is making of semiconductor wafer , even microscopic dents or cracks on silicon wafer surfaces can lead to yield loss and electrical failure during chip fabrication.  
Manual inspection under microscopes is **slow, subjective, and prone to human error**.  
The challenge is to **automate quality control (QC)** using **image processing** that can detect such surface defects reliably at the pre-fabrication stage.

## 💡 Solution Overview

We propose an **Automated grayscale analysis** system that detects **microcracks and dents** on polished silicon wafer surfaces using standard camera images.  

Our system:
- Uses **high-resolution images** of the wafer under controlled lighting which are fed as input.
- Converts them to **grayscale intensity maps** for reflection uniformity analysis.
- Uses **OpenCV and scikit-image** algorithms (thresholding, edge detection) to highlight surface defects in order to distinctly seperate them from the undefected areas. 

The output is a processed image where **defect regions are automatically highlighted** and a basic report is generated.

## ⚙️ Tech Stack

**Programming Language**
- Python 3.x

**Libraries**
- OpenCV (cv2) — Image processing & feature extraction  
- scikit-image — Thresholding, segmentation & texture analysis  
- NumPy — Matrix operations on pixel arrays  
- scikit-learn — Simple classification (optional)  
- Matplotlib / Seaborn — Visualization  
- Jupyter Notebook — For debugging & stepwise visualization

## System Architecture

[Architecture Diagram](architecture_diagram.png)

**Flow Explanation:**
1. Image Capture (DSLR or industrial camera under uniform light)  
2. Grayscale Conversion & Denoising  
3. Thresholding / Edge Detection  
4. Morphological Refinement (to isolate defects)  
5. Feature Extraction (edge density, shape, texture)  
6. Classification (defect type prediction – optional)  
7. Result Visualization & Reporting  

## 🧰 Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Dhwani236/Dquad_HackX
   cd Dquad_HackX

2. **Create and Activate a Virtual Environment**
   ```bash
      python -m venv venv
      venv\Scripts\activate 

4. **Install dependencies**
   ```bash
    pip install -r requirements.txt

6. **Run the prototype**
   ```bash
    cd src
    python main.py


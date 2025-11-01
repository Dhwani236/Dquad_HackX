"""
prototype.py
============

Grayscale wafer-surface QC prototype:
- Dark/flat correction
- Preprocessing (bg removal, CLAHE, denoise)
- Crack detection (Canny + Frangi ridge enhancement)
- Dent detection (LoG blob detection)
- Feature extraction per candidate region
- Train/predict RandomForest classifier (optional)
- Visualization overlays and CSV output

Usage (examples):
    python prototype.py --input_dir data/images --out_dir outputs
    python prototype.py --input_dir data/images --out_dir outputs --train_csv labels.csv

Notes:
- Images: support 8/16-bit single-channel or color TIFF/PNG/JPG (color converted to grayscale).
- Labels CSV format (optional for training): filename, x, y, w, h, label
  where label in {"crack","dent","other"} and bbox coordinates in pixels.
"""

import os
import argparse
import glob
import csv
from pathlib import Path

import numpy as np
import cv2
from skimage import filters, feature, morphology, exposure, util
from skimage.feature import local_binary_pattern, blob_log
from skimage.filters import frangi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# Utility IO functions
# -------------------------
def imread_any(path):
    """Read image preserving bit depth using OpenCV; convert color -> grayscale if needed."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # If color, convert BGR->Gray using luminosity
    if img.ndim == 3:
        b, g, r = cv2.split(img)
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(img.dtype)
        return gray
    return img

def save_overlay(out_path, img_gray, mask_cracks, mask_dents, contours_cracks):
    """Save visualization overlay (PNG)."""
    # normalize to 8-bit for visualization
    if img_gray.dtype != np.uint8:
        img_vis = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_vis = img_gray.copy()
    color = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    # cracks in red
    color[mask_cracks > 0] = (0, 0, 255)
    # dents in blue
    color[mask_dents > 0] = (255, 0, 0)
    # draw crack contours
    cv2.drawContours(color, contours_cracks, -1, (0,255,255), 1)
    cv2.imwrite(str(out_path), color)

# -------------------------
# Dark & flat field correction
# -------------------------
def dark_flat_correction(img, dark=None, flat=None):
    """Perform dark-frame subtraction and flat-field division.
    Inputs/outputs are float32 arrays.
    """
    img_f = img.astype(np.float32)
    if dark is not None:
        dark_f = dark.astype(np.float32)
        img_f = img_f - dark_f
    if flat is not None:
        flat_f = flat.astype(np.float32)
        # avoid division by 0
        denom = flat_f - (dark.astype(np.float32) if dark is not None else 0.0)
        denom = np.clip(denom, a_min=1e-6, a_max=None)
        img_f = img_f / denom
        # normalize
        img_f = img_f - np.min(img_f)
    # final normalize to full range
    img_f = img_f - img_f.min()
    if img_f.max() > 0:
        img_f = img_f / img_f.max()
    return img_f

# -------------------------
# Preprocessing
# -------------------------
def preprocess(img_f, large_blur_sigma=150, clahe_clip=0.01, denoise_med=3):
    """img_f expected float in [0,1]. Returns 8-bit grayscale."""
    # remove large-scale illumination (background)
    bg = cv2.GaussianBlur((img_f * 65535).astype(np.uint16), (0,0), sigmaX=large_blur_sigma, sigmaY=large_blur_sigma)
    bg = bg.astype(np.float32) / 65535.0
    img_flat = img_f - bg
    # avoid negative
    img_flat = img_flat - img_flat.min()
    if img_flat.max() > 0:
        img_flat = img_flat / img_flat.max()
    # contrast limited adaptive histogram equalization (CLAHE) - use skimage exposure
    img_uint8 = (img_flat * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_uint8)
    # denoise: median then bilateral for edge preservation
    img_med = cv2.medianBlur(img_clahe, denoise_med)
    img_bilat = cv2.bilateralFilter(img_med, d=5, sigmaColor=75, sigmaSpace=75)
    return img_bilat

# -------------------------
# Crack / scratch detection
# -------------------------
def detect_cracks(img_u8, canny_sigma=1.0, canny_low=50, canny_high=150, frangi_scales=(1,2,3)):
    """Return binary mask for crack candidates and list of contours."""
    # Canny edges
    edges = cv2.Canny(img_u8, canny_low, canny_high, L2gradient=True)
    # Frangi filter to enhance ridges (expects float)
    img_float = img_u8.astype(np.float32) / 255.0
    fr = frangi(img_float, sigmas=frangi_scales, scale_range=None, scale_step=None)
    # threshold frangi response adaptively
    thr = np.nan_to_num(fr) > (np.mean(fr) + 0.5*np.std(fr))
    # combine edges and frangi
    comb = ((edges > 0).astype(np.uint8) | thr.astype(np.uint8)).astype(np.uint8)
    # morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    comb = cv2.morphologyEx(comb*255, cv2.MORPH_CLOSE, kernel, iterations=1)
    comb = (comb>0).astype(np.uint8)
    # skeletonize to reduce to 1-pixel wide
    comb_skel = morphology.skeletonize(comb > 0).astype(np.uint8)
    # find contours
    contours, _ = cv2.findContours((comb_skel*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # remove very small contours
    good_contours = [c for c in contours if cv2.arcLength(c, closed=False) > 10]  # tunable
    mask = np.zeros_like(img_u8, dtype=np.uint8)
    cv2.drawContours(mask, good_contours, -1, 255, thickness=1)
    return mask, good_contours

# -------------------------
# Dent / pit detection (blob)
# -------------------------
def detect_dents(img_u8, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.02):
    """Use Laplacian of Gaussian blob detection to find dents/pits.
    Returns binary mask and list of blobs (y,x,radius)
    """
    img_float = img_u8.astype(np.float32) / 255.0
    blobs = blob_log(img_float, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    # Compute radii in pixels for each blob (sqrt(2)*sigma)
    blobs[:,2] = blobs[:,2] * np.sqrt(2)
    mask = np.zeros_like(img_u8, dtype=np.uint8)
    for y, x, r in blobs:
        cv2.circle(mask, (int(x), int(y)), int(max(1, r)), 255, thickness=-1)
    return mask, blobs

# -------------------------
# Feature extraction for ML
# -------------------------
def extract_features(img_u8, mask, contours=None, blobs=None):
    """Extract features for each candidate region. Returns list of dicts."""
    feats = []
    img_float = (img_u8.astype(np.float32) / 255.0)
    # regions from contours
    if contours:
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area < 5:
                continue
            roi = img_u8[y:y+h, x:x+w]
            # geometry
            aspect = w/h if h>0 else 0
            # intensity stats
            mean_int = float(np.mean(roi))
            std_int = float(np.std(roi))
            # edge density via Canny
            edges = cv2.Canny(roi, 50,150)
            edge_density = float(edges.mean())
            # LBP texture histogram (uniform LBP)
            lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0,10))
            hist = hist.astype("float")
            if hist.sum() > 0:
                hist /= hist.sum()
            feat = {
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
                'aspect': aspect, 'mean_int': mean_int, 'std_int': std_int,
                'edge_density': edge_density
            }
            # add lbp bins
            for i, val in enumerate(hist):
                feat[f'lbp_{i}'] = float(val)
            feats.append(feat)
    # regions from blobs
    if blobs is not None:
        for blob in blobs:
            y,x,r = blob
            x = int(x); y=int(y); r=int(max(1, r))
            x0 = max(0, x-r); y0 = max(0, y-r); x1 = min(img_u8.shape[1], x+r); y1 = min(img_u8.shape[0], y+r)
            roi = img_u8[y0:y1, x0:x1]
            area = np.pi * r * r
            mean_int = float(np.mean(roi)) if roi.size>0 else 0.0
            std_int = float(np.std(roi)) if roi.size>0 else 0.0
            feat = {'x': x0, 'y': y0, 'w': x1-x0, 'h': y1-y0, 'area': area,
                    'aspect': 1.0, 'mean_int': mean_int, 'std_int': std_int, 'edge_density': 0.0}
            # add zeros for lbp fields to keep feature shape consistent
            for i in range(10):
                feat[f'lbp_{i}'] = 0.0
            feats.append(feat)
    return feats

# -------------------------
# Train classifier (optional)
# -------------------------
def train_classifier_from_csv(csv_labels, images_dir):
    """
    CSV format:
    filename,x,y,w,h,label
    label: crack / dent / other
    """
    X = []
    y = []
    label_map = {'crack':0, 'dent':1, 'other':2}
    with open(csv_labels, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fname = os.path.join(images_dir, row['filename'])
            if not os.path.exists(fname):
                continue
            img = imread_any(fname)
            # convert to normalized float & basic preprocessing
            img_f = img.astype(np.float32)
            img_f = img_f - img_f.min()
            if img_f.max()>0: img_f /= img_f.max()
            img_u8 = (img_f * 255).astype(np.uint8)
            # cut ROI
            x = int(row['x']); y=int(row['y']); w=int(row['w']); h=int(row['h'])
            roi = img_u8[y:y+h, x:x+w]
            # extract simple features: area,w,h, mean,std, edge density, lbp hist
            area = w*h
            mean_int = float(np.mean(roi)) if roi.size>0 else 0.0
            std_int = float(np.std(roi)) if roi.size>0 else 0.0
            edges = cv2.Canny(roi, 50,150) if roi.size>0 else np.zeros((1,1),dtype=np.uint8)
            edge_density = float(edges.mean())
            lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0,10))
            hist = hist.astype("float")
            if hist.sum() > 0: hist /= hist.sum()
            feat = [area, w/h if h>0 else 0, mean_int, std_int, edge_density]
            feat += hist.tolist()
            X.append(feat)
            y.append(label_map.get(row['label'].strip().lower(), 2))
    if len(X) < 10:
        print("[train_classifier] Not enough labeled samples for training (need >=10). Skipping training.")
        return None
    X = np.array(X); y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    print("Validation report:\n", classification_report(y_val, preds))
    return clf

# -------------------------
# Prediction & visualization pipeline for a single image
# -------------------------
def process_image(path, dark=None, flat=None, classifier=None, out_dir=None):
    img = imread_any(path)
    img_name = Path(path).stem
    # convert to float and dark/flat correct
    img_f = img.astype(np.float32)
    if dark is not None or flat is not None:
        dark_img = imread_any(dark) if dark else None
        flat_img = imread_any(flat) if flat else None
        img_f = dark_flat_correction(img, dark_img if dark else None, flat_img if flat else None)
    else:
        # normalize if no correction
        img_f = img_f - img_f.min()
        if img_f.max() > 0:
            img_f = img_f / img_f.max()
    # preprocess -> 8-bit image
    img_u8 = preprocess(img_f)
    # detect cracks
    mask_cracks, contours = detect_cracks(img_u8)
    # detect dents
    mask_dents, blobs = detect_dents(img_u8)
    # features
    feats = extract_features(img_u8, mask_cracks, contours, blobs)
    # prepare classification if model present
    predictions = []
    if classifier is not None and feats:
        # create consistent feature vector ordering
        feature_names = ['area','aspect','mean_int','std_int','edge_density'] + [f'lbp_{i}' for i in range(10)]
        X = []
        for f in feats:
            vec = [f.get(n, 0.0) for n in feature_names]
            X.append(vec)
        preds = classifier.predict(X)
        predictions = preds.tolist()
    # Save overlay and CSV of defects
    out_dir = Path(out_dir) if out_dir else Path('./outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{img_name}_overlay.png"
    save_overlay(overlay_path, (img_f*255).astype(np.uint8), mask_cracks, mask_dents, contours)
    # write defects.csv
    csv_path = out_dir / f"{img_name}_defects.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['x','y','w','h','area','aspect','mean_int','std_int','edge_density','pred_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, f in enumerate(feats):
            row = {k: f.get(k,0) for k in ['x','y','w','h','area','aspect','mean_int','std_int','edge_density']}
            row['pred_label'] = (predictions[i] if predictions else '')
            writer.writerow(row)
    print(f"[process_image] processed {path}. overlay -> {overlay_path}, defects csv -> {csv_path}")
    return overlay_path, csv_path

# -------------------------
# CLI & main
# -------------------------
def main(args):
    # load optional model or train if csv provided
    clf = None
    if args.train_csv:
        clf = train_classifier_from_csv(args.train_csv, args.input_dir)
        if clf and args.save_model:
            dump(clf, args.save_model)
            print(f"[main] saved model to {args.save_model}")
    elif args.model:
        clf = load(args.model)
        print(f"[main] loaded model {args.model}")

    # optional dark & flat
    dark = args.dark if args.dark and os.path.exists(args.dark) else None
    flat = args.flat if args.flat and os.path.exists(args.flat) else None

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))
    for p in tqdm(image_paths):
        ext = Path(p).suffix.lower()
        if ext not in ['.tif','.tiff','.png','.jpg','.jpeg','bmp']:
            continue
        process_image(p, dark=dark, flat=flat, classifier=clf, out_dir=args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wafer grayscale QC prototype")
    parser.add_argument('--input_dir', type=str, required=True, help='Input image folder')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Output folder for overlays/csv')
    parser.add_argument('--dark', type=str, default=None, help='Dark frame image path (same exposure)')
    parser.add_argument('--flat', type=str, default=None, help='Flat-field image path (uniform illumination)')
    parser.add_argument('--train_csv', type=str, default=None, help='CSV labels to train classifier (optional)')
    parser.add_argument('--model', type=str, default=None, help='Load pretrained joblib model')
    parser.add_argument('--save_model', type=str, default='rf_model.joblib', help='Save trained model')
    args = parser.parse_args()
    main(args)

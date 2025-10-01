import cv2 
import numpy as np
from pathlib import Path
import json

# --------------------------- Paths ---------------------------
dataset_dir = Path("/Users/mohammadbilal/Documents/Projects/STM32-InstanceSegmentation/base_model/Dataset")
results_dir = Path("/Users/mohammadbilal/Documents/Projects/STM32-InstanceSegmentation/base_model/results")

images_out_dir = results_dir / "images"
labels_out_dir = results_dir / "labels"
overlay_out_dir = results_dir / "overlay"
images_out_dir.mkdir(parents=True, exist_ok=True)
labels_out_dir.mkdir(parents=True, exist_ok=True)
overlay_out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------- Helper Functions ---------------------------
def resize(img, perc=0.5):
    width = int(img.shape[1] * perc)
    height = int(img.shape[0] * perc)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def dark_channel(img, sz=15):
    b, g, r = cv2.split(img)
    dc = r - cv2.max(b, g)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.erode(dc, kernel)

def airlight(img, dc):
    h, w = img.shape[:2]
    imgvec = img.reshape(h*w, 3)
    dcvec = dc.reshape(h*w)
    indices = dcvec.argsort()
    return imgvec[indices[0]]

def transmission_estimate(dc):
    return dc + (1 - np.max(dc))

def guided_filter(img, p, r, eps):
    mean_I = cv2.boxFilter(img, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(img * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(img * img, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * img + mean_b

def transmission_refine(img, et):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    return guided_filter(gray, et, r, eps)

def recover(img, t, A):
    res = np.empty(img.shape, img.dtype)
    for i in range(3):
        res[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]
    return res

def normalize_image(img):
    img = img - img.min()
    img = img / img.max() * 255
    return np.uint8(img)

def edge_enhancement(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(gray)

    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    edges = cv2.Canny(img, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[edges > 0] = [0, 0, 255]

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    return mask

# --------------------------- Process Dataset ---------------------------
scale = 1
min_area = 1000
img_counter = 1
class_mapping = {}  # Dictionary to store subfolder name -> class ID
class_id = 0

for subfolder in sorted(dataset_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    print(f"Processing folder: {subfolder.name}")
    class_mapping[subfolder.name] = class_id  # Assign class ID to this subfolder
    
    for img_file in sorted(subfolder.glob("*.png")):
        print(f"Processing: {img_file.name}")
        src = cv2.imread(str(img_file))
        if src is None:
            continue
        
        src = resize(src, scale)
        img = src.astype('float64') / 255

        dc = dark_channel(img, 15)
        te = transmission_estimate(dc)
        tr = transmission_refine(src, te)
        A = airlight(img, tr)
        result = recover(img, tr, A)
        result = normalize_image(result)
        mask = edge_enhancement(result)

        # Clean mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                clean_mask[labels == i] = 255

        organism = cv2.bitwise_and(result, result, mask=clean_mask)

        # Save processed image
        out_img_name = f"{img_counter}.png"
        cv2.imwrite(str(images_out_dir / out_img_name), result)

        # Generate YOLO polygon labels
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_labels = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygon = []
            h, w = clean_mask.shape[:2]
            for point in largest_contour:
                x, y = point[0]
                polygon.extend([x / w, y / h])
            yolo_labels.append([class_id] + polygon)

        # Save label file
        label_file = labels_out_dir / f"{img_counter}.txt"
        with open(label_file, "w") as f:
            for label in yolo_labels:
                f.write(" ".join(map(str, label)) + "\n")

        # Overlay
        overlay = src.copy()
        if contours:
            cv2.drawContours(overlay, [largest_contour], -1, (0,255,0), 2)
        cv2.imwrite(str(overlay_out_dir / out_img_name), overlay)

        img_counter += 1

    class_id += 1  # Increment class ID for next subfolder

# Save class mapping dictionary to text file
mapping_file = results_dir / "class_mapping.txt"
with open(mapping_file, "w") as f:
    for k, v in class_mapping.items():
        f.write(f"{k}:{v}\n")

print("Processing complete!")
print(f"Class mapping saved to {mapping_file}")
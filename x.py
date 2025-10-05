import shutil
from pathlib import Path
import random

# Paths
results_dir = Path("/Users/mohammadbilal/Documents/Projects/STM32-InstanceSegmentation/base_model/results")
dataset_dir = Path("/Users/mohammadbilal/Documents/Projects/STM32-InstanceSegmentation/base_model/yolo_dataset")

# Train/Val split ratio
split_ratio = 0.8  

# Create folders
for split in ["train", "val"]:
    (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Collect all images
images = sorted((results_dir / "images").glob("*.png"))

random.shuffle(images)
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
val_images = images[split_index:]

def move_files(image_list, split):
    for img in image_list:
        label = results_dir / "labels" / (img.stem + ".txt")
        if label.exists():
            shutil.copy(img, dataset_dir / "images" / split / img.name)
            shutil.copy(label, dataset_dir / "labels" / split / (img.stem + ".txt"))

move_files(train_images, "train")
move_files(val_images, "val")

print("Dataset structured for YOLO instance segmentation!")
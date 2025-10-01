from ultralytics import YOLO
import os

# YOLOv8n-seg is lightweight and suitable for embedded deployment
model = YOLO("base_model/models/yolov8n-seg.pt")  # pretrained on COCO
dataset_yaml = "base_model/dataset/data.yaml"

train_params = {
    "data": dataset_yaml,
    "epochs": 70,           # adjust if needed
    "imgsz": 640,           # image size for training
    "device": "mps",        # or "0" for GPU
    "batch": 8,             # adjust according to RAM/GPU
    "augment": True,        # enable augmentation
    "patience": 20,         # early stopping patience
    "verbose": True
}

results = model.train(**train_params)

metrics = model.val()
print("Validation metrics:", metrics)

final_model_path = "yolov8n_seg_plankton.pt"
model.save(final_model_path)
print(f"Final YOLOv8 model saved as: {final_model_path}")

"""
test_image = "test/images/12_prorocentrum_micans.jpg"
prediction = model(test_image)
prediction[0].show()  # display results
"""

# Export to ONNX for STM32 Deployment
onnx_path = model.export(format="onnx", opset=17, dynamic=True)
print("ONNX model saved at:", onnx_path)
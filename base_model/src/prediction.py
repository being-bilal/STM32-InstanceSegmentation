from ultralytics import YOLO
import os

model = YOLO("base_model/models/yolov8n-seg.pt")  # pretrained on COCO
dataset_yaml = "/Users/mohammadbilal/Documents/Projects/STM32-InstanceSegmentation/base_model/yolo_dataset/data.yaml"

train_params = {
    "data": dataset_yaml,
    "epochs": 20,           
    "imgsz": 640,           
    "device": "mps",        
    "batch": 16,           
    "augment": True, # enable augmentation
    "patience": 5,        
    "verbose": True
}

results = model.train(**train_params)

metrics = model.val()
print("Validation metrics:", metrics)

final_model_path = "yolov8n_seg_plankton.pt"
model.save(final_model_path)

"""
test_image = "test/images/12_prorocentrum_micans.jpg"
prediction = model(test_image)
prediction[0].show()  # display results
"""

# Export to ONNX for STM32 Deployment
onnx_path = model.export(format="onnx", opset=17, dynamic=True)
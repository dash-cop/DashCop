from ultralytics import YOLO
import wandb

# Start the training from a pretrained segmentation model 
model = YOLO("yolov8n-seg.pt")

results = model.train(data="cfg.yaml", epochs=100, imgsz=640, device='0')
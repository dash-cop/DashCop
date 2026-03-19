

'''
PYTHONPATH="/home/sai.teja/DashCop/training/SAC:$PYTHONPATH" python train.py
use the above command to run for multiple gpus  
'''


import sys
import os


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


import sys
sys.path.append("/home/sai.teja/DashCop/training/SAC/ultralytics")


from ultralytics import YOLO
import ultralytics

# Disable wandb in settings before training
from ultralytics.utils import SETTINGS
SETTINGS['wandb'] = False

# print("ultralytics module path:")
# print(ultralytics.__file__)

# exit(0)

# import wandb

# Start the training from a pretrained segmentation model 
model = YOLO("yolov8n-seg.pt")

results = model.train(
    data="cfg.yaml",
    epochs=200,
    imgsz=640,
    device=[0,1],
    lr0=0.01,
    batch=128,
    project="SAC_models",
    name="lr0p01",
    save_dir="/ssd_scratch/sai.teja/SAC_models"
)

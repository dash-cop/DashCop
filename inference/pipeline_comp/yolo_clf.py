import sys
# sys.path.append('/home2/keshav06/.local/lib/python3.6/site-packages')
import os
import shutil
import time
from PIL import Image
import cv2
import numpy as np
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
# from instance_funcs import *
from ultralytics import YOLO
# from core.association import *
from transformers import ViTImageProcessor, ViTModel
import matplotlib.pyplot as plt
import joblib
import torch.nn as nn
from pipeline_comp.models.load import *
from pipeline_comp.models.lit_model import Model

class YOLOClf():
    def __init__(self, weights_path, rm_assoc_path, num_classes=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self.transforms()

        model = give_yolo_model(weights=rm_assoc_path, num_classes=4)
        model = Model(model, None, 4)
        model_to_load = torch.load(weights_path)
        model.load_state_dict(model_to_load['state_dict'])
        model.to("cuda")
        model.eval()
        
        self.model = model
    
    def transforms(self):
        return transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def __call__(self, x):
        x = Image.fromarray(x)
        crop = self.transform(x)[None].to(self.device)
        out_pred = torch.argmax(self.model(crop)[0]).item() + 1
        print(out_pred)
        if(out_pred == 4):
            out_pred = 0
        return out_pred
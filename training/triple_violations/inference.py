import sys
import os
import shutil
import time
from PIL import Image
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from ultralytics import YOLO
import matplotlib.pyplot as plt
from yolo_clf import YOLOClf


assoc_weights = "/archive/sai.teja/DashCop_ckpts/rm_assoc_model/model_ft.pt"
clf_weights = "/archive/sai.teja/DashCop_ckpts/tr_checkpoints/tr_clf.ckpt"

dt_classifier = YOLOClf(weights_path=clf_weights, rm_assoc_path=assoc_weights)

image_path = "/ssd_scratch/sai.teja/bal_tr_dataset_ogsplit/val/2_double/20211125132806_0060_frame001330_motor1009.jpg"
img = cv2.imread(image_path)

# pred = dt_classifier(img)
# print(f"prediction from the model: {pred}")

image_paths = glob("/ssd_scratch/sai.teja/bal_tr_dataset_ogsplit/val/3_triple/*.jpg")
output_dir = "./checking_tr_clf_p1/3_triple"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image_path in tqdm(image_paths,desc = "Processing images"):
    img = cv2.imread(image_path)
    pred = dt_classifier(img) + 1

    # save the image with the prediction on the top right of the image
    cv2.putText(img, str(pred), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img)
    


import sys
sys.path.append('../')
sys.path.append('tracker')
sys.path.append('/home2/keshav06/.local/lib/python3.6/site-packages')
import os
import shutil
import time
from absl import app, flags, logging
from absl.flags import FLAGS
# from core.association import *
# from core.config import cfg
from PIL import Image
import cv2
import numpy as np
# import pickle
import torch
import matplotlib.pyplot as plt
from instance_funcs import *

from ultralytics import YOLO
from core.association import *
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, DPTForDepthEstimation
import glob
from transformers import VisionEncoderDecoderModel, TrOCRForCausalLM
from transformers import TrOCRProcessor
from pipeline_comp.det_assoc import DetAssoc
from pipeline_comp.yolo_clf import MAEClf

imgs = glob.glob("/ssd_scratch/cvit/keshav/test/inst_crops_single_double/*.png")
dt_classifier = MAEClf(clf_model_path='../tr_exp/model.pkl', feature_extractor_path='/ssd_scratch/cvit/keshav/vit_mae_rider')
ones = 0
zeros = 0
for img in imgs:
    img = np.asarray(Image.open(img))[:, :, :3]
    out = dt_classifier(img)[0]
    if(out == 1):
        ones += 1
    else:
        zeros += 1

print(ones)
print(zeros)

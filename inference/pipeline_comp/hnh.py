import sys
# sys.path.append('/home2/keshav06/.local/lib/python3.6/site-packages')
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
# from core.association import *

class HNHModel():
    def __init__(self, weights_path, conf_score=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights_path)
        self.conf_score = conf_score

    def format_boxes_xyxy(self, bboxes, image_height, image_width):
        for box in bboxes:
            ymin = int(box[1] * image_height)
            xmin = int(box[0] * image_width)
            ymax = int(box[3] * image_height)
            xmax = int(box[2] * image_width)
            box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
        return bboxes

    def format_boxes_xyxy2xywh(self, bboxes):
        for box in bboxes:
            ymin = int(box[1])
            xmin = int(box[0])
            ymax = int(box[3])
            xmax = int(box[2])
            width = xmax - xmin
            height = ymax - ymin
            box[0], box[1], box[2], box[3] = int((xmin + xmax) / 2), int((ymin + ymax) / 2), width, height
        return bboxes

    def __call__(self, frame):
        '''
        preds is a YOLOv8 object 
        frame is the img of the detection 
        '''
        preds = self.model(frame, classes = [0, 1], conf = self.conf_score)[0]
        boxes = preds.boxes
        boxes_hnh = boxes.xyxy.detach().cpu().numpy()
        
        # Normalize the bounding boxes
        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        boxes_hnh[:, 0::2] /= frame.shape[1]
        boxes_hnh[:, 1::2] /= frame.shape[0]

        # Get the scores and classes
        scores_hnh = boxes.conf.detach().cpu().numpy()
        classes_hnh = boxes.cls.detach().cpu().numpy()
        classes_hnh = classes_hnh[:]

        bboxes = boxes_hnh
        scores = scores_hnh
        classes = classes_hnh
        
        original_h, original_w, _ = frame.shape
        bboxes = self.format_boxes_xyxy(bboxes, original_h, original_w)

        return bboxes, scores, classes
import sys
# sys.path.append('/home2/keshav06/.local/lib/python3.6/site-packages')
from PIL import Image
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
# from instance_funcs import *
from ultralytics import YOLO
# from core.association import *
import matplotlib.pyplot as plt

class DetAssoc():
    def __init__(self, weights_path, conf_score=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights_path)
        # self.model.to(self.get_freest_cuda_device())
        self.model.to("cuda:0")
        print(self.model.device)
        self.conf_score = conf_score
    
    def get_freest_cuda_device(self):
        device_count = torch.cuda.device_count()
        free_memory = [torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i) for i in range(device_count)]
        most_free_device = free_memory.index(max(free_memory))
        return most_free_device

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

    def getDetections(self, preds, frame):
        '''
        preds is a YOLOv8 object 
        frame is the img of the detection 
        '''
        rider_motor_bboxes = preds.boxes
        boxes_rm = rider_motor_bboxes.xyxy.detach().cpu().numpy()
        
        # Normalize the bounding boxes
        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        boxes_rm[:, 0::2] /= frame.shape[1]
        boxes_rm[:, 1::2] /= frame.shape[0]

        # Get the scores and classes
        scores_rm = rider_motor_bboxes.conf.detach().cpu().numpy()
        classes_rm = rider_motor_bboxes.cls.detach().cpu().numpy()
        classes_rm = classes_rm[:]

        bboxes = boxes_rm
        scores = scores_rm
        classes = classes_rm

        return bboxes, scores, classes
    
    def compute_mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou


    def remove_duplicate_masks(self, masks):
        indices_to_remove = set()
        n_masks = masks.shape[0]

        for i in range(n_masks):
            for j in range(i + 1, n_masks):
                iou = self.compute_mask_iou(masks[i], masks[j])
                if iou > 0.7:
                    if np.all(masks[i] == masks[j]):
                        indices_to_remove.add(j)
                    elif np.all(masks[i] <= masks[j]):
                        indices_to_remove.add(i)
                    else:
                        indices_to_remove.add(j)

        masks_filtered = np.delete(masks, list(indices_to_remove), axis=0)
        return masks_filtered, list(indices_to_remove)

    def __call__(self, frame):

        preds = self.model(frame, classes = [0, 1], conf = self.conf_score)[0]
        if(len(preds) == 0):
           return [], [], []
        
        # Get the detections
        rm_boxes, rm_scores, rm_classes = self.getDetections(preds, frame)

        rider_boxes = rm_boxes[rm_classes == 0]
        rider_scores = rm_scores[rm_classes == 0]
        rider_classes = rm_classes[rm_classes == 0]

        motor_boxes = rm_boxes[rm_classes == 1]
        motor_scores = rm_scores[rm_classes == 1]
        motor_classes = rm_classes[rm_classes == 1]
        
        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        original_h, original_w, _ = frame.shape

        rider_boxes = self.format_boxes_xyxy(rider_boxes, original_h, original_w)
        motor_boxes = self.format_boxes_xyxy(motor_boxes, original_h, original_w)
        
        rider_masks_ = preds.masks.data[rm_classes == 0].cpu().numpy()
        motor_masks_ = preds.masks.data[rm_classes == 1].cpu().numpy()
        rider_cross_masks_ = preds.masks_cross.data[rm_classes == 0].cpu().numpy()
        motor_cross_masks_ = preds.masks_cross.data[rm_classes == 1].cpu().numpy()

        rider_masks_, inds = self.remove_duplicate_masks(rider_masks_)
        rider_boxes = np.delete(rider_boxes, inds, 0)
        rider_scores = np.delete(rider_scores, inds, 0)
        rider_classes = np.delete(rider_classes, inds, 0)
        rider_cross_masks_ = np.delete(rider_cross_masks_, inds, 0)

        motor_masks_, inds = self.remove_duplicate_masks(motor_masks_)
        motor_boxes = np.delete(motor_boxes, inds, 0)
        motor_scores = np.delete(motor_scores, inds, 0)
        motor_classes = np.delete(motor_classes, inds, 0)
        motor_cross_masks_ = np.delete(motor_cross_masks_, inds, 0)
        
        rider_masks = []
        motor_masks = []
        rider_cross_masks = []
        motor_cross_masks = []
        for i in range(len(rider_masks_)):
            rider_masks.append(cv2.resize(rider_masks_[i], (160, 93)))
            rider_cross_masks.append(cv2.resize(rider_cross_masks_[i], (160, 93)))
        for i in range(len(motor_masks_)):
            motor_masks.append(cv2.resize(motor_masks_[i], (160, 93)))
            motor_cross_masks.append(cv2.resize(motor_cross_masks_[i], (160, 93)))

        rider_masks = np.array(rider_masks)
        motor_masks = np.array(motor_masks)
        rider_cross_masks = np.array(rider_cross_masks)
        motor_cross_masks = np.array(motor_cross_masks)

        rider_classes = rider_classes.astype(np.int32)
        motor_classes = motor_classes.astype(np.int32)

        # The tracker will accept boxes in the format (xc, yc, w, h)
        # rider_boxes = format_boxes_xyxy2xywh(rider_boxes)
        rider_dets = np.column_stack((rider_boxes, rider_scores, rider_classes))
        motor_dets = np.column_stack((motor_boxes, motor_scores, motor_classes))
        all_preds_data = np.concatenate([rider_dets, motor_dets])
        try:
            all_masks = np.concatenate([rider_masks, motor_masks], 0)
            all_cross_masks = np.concatenate([rider_cross_masks, motor_cross_masks], 0)
        except:
            if(len(motor_masks) == 0):
                all_masks = rider_masks
                all_cross_masks = rider_cross_masks
            else:
                all_masks = motor_masks
                all_cross_masks = motor_cross_masks

        return all_preds_data, all_masks, all_cross_masks
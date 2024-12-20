import torch
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import numpy as np
import cv2

class LicensePlateDet():
    def __init__(self, weights_path, conf_score=0.5):
        self.det_weights_path = weights_path
        self.conf_score = conf_score
        self.model = YOLO(weights_path)

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
        preds = self.model(frame, conf = self.conf_score)[0]
        boxes = preds.boxes
        boxes_lp = boxes.xyxy.detach().cpu().numpy()
        
        # Normalize the bounding boxes
        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        boxes_lp[:, 0::2] /= frame.shape[1]
        boxes_lp[:, 1::2] /= frame.shape[0]

        # Get the scores and classes
        scores_lp = boxes.conf.detach().cpu().numpy()
        classes_lp = boxes.cls.detach().cpu().numpy()
        classes_lp = classes_lp[:]

        bboxes = boxes_lp
        scores = scores_lp
        classes = classes_lp
        
        original_h, original_w, _ = frame.shape
        bboxes = self.format_boxes_xyxy(bboxes, original_h, original_w)

        return bboxes, scores, classes

class LicensePlateRec():
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained(weights_path, token='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')
        self.model.to(self.device)

    def __call__(self, crop):
        pixel_values = self.processor(crop, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

class LicensePlateModel():
    def __init__(self, det_weights, rec_weights, det_only=False):
        self.detector = LicensePlateDet(det_weights)
        if(not det_only):
            self.recognizor = LicensePlateRec(rec_weights)
        else:
            self.recognizor = None
    
    def __call__(self, frame):
        # Detect First
        bboxes, scores, _ = self.detector(frame)
        # Recognize the detections
        lp_numbers = []
        for box in bboxes:
            l, t, r, b = box.astype(np.int32)
            crop = frame[t:b, l:r, :]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if(self.recognizor is not None):
                lp_num = self.recognizor(crop)
            else:
                lp_num = None
            lp_numbers.append(lp_num)
        
        return bboxes, lp_numbers

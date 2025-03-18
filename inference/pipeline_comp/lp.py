import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import sys
sys.path.append('/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg/modules/ms_deform_attn.py')
sys.path.append('/home2/keshav06/TrafficViolations/deep-text-recognition-benchmark/modules')
sys.path.append('/home2/deepti.rawat/space_issue/home/keshav/deep-text-recognition-benchmark/')
# sys.path.insert(0, '/home2/keshav06/TrafficViolations/deep-text-recognition-benchmark')
from .lp_utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import os
import torchvision.transforms as transforms
from PIL import Image
import cv2
# import lp_utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from ultralytics import YOLO
import numpy as np
import cv2

def deskew_plate(image):
    minangle = 0
    maxangle = 0
    min_area = 30*30
    average_h = 50
    image2 = image.copy()
    # print(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # use thresholding so that grey pixels are turned white and black pixels are turned black
    _, grayscale = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)


    # Find the contours in the image
    contours, hierarchy = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # print the contour
    # cv2.drawContours(image2, [max_contour], 0, (0, 255, 0), 3)
    # find the angle from the horizontal line of the contour and rotate the image
    try:
        rect = cv2.minAreaRect(max_contour)
    except:
        print(f"Skipping as no contour found")
        return None
    # crop the image to the bounding box of the contour from up and down
    top_most_point = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    bottom_most_point = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
    left_most_point = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    right_most_point = tuple(max_contour[max_contour[:, :, 0].argmax()][0])

    cropped_image = image[top_most_point[1]:bottom_most_point[1], left_most_point[0]:right_most_point[0]]
    image = cropped_image

    # print(rect)
    angle = rect[2]
    minangle = min(minangle, angle)
    maxangle = max(maxangle, angle)
    # angle is between 0 and 90, with 45 being the horizontal line
    if angle < 45: # if angle is less than 45, then it is the angle from the horizontal line to the right
        angle = angle
    else: # if angle is greater than 45, then it is the angle from the horizontal line to the left
        angle = angle - 90
    # print(angle)
    if image.shape[0] * image.shape[1] < min_area:
        print(f"Skipping as area is too small")
        return None
    image_height = image.shape[0]
    image_width = image.shape[1]
    if image_height > image_width or image_width/image_height > 2.5:
        print(f"Skipping as aspect ratio is too high")
        return None
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D(rect[0], angle, 1), (image.shape[1], image.shape[0]))
    # cv2.imwrite(os.path.join(output_folder, file), rotated)
    # save grayscale image
    # cv2.imwrite(os.path.join(output_folder, file.replace('.png', '_contour.png')), image2)

    # increase the height of the upper half to have 3/5ths of the top half
    upper = rotated[:rotated.shape[0]* 3//5, :]
    lower = rotated[rotated.shape[0] * 2//5:, :]
    # ensure that both have same number of rows
    if upper.shape[0] != lower.shape[0]:
        lower = cv2.resize(lower, (upper.shape[1], upper.shape[0]))
    concatenated = cv2.hconcat([upper, lower])

    # image aspect ratio
    h = concatenated.shape[0]
    w = concatenated.shape[1]
    aspect_ratio = w/h
    if h < average_h:
    # resize the concatenated to have a height of average_h, bicubic interpolation
        concatenated = cv2.resize(concatenated, (int(average_h*aspect_ratio), average_h), interpolation=cv2.INTER_CUBIC)
    return concatenated

class LicensePlateDet():
    def __init__(self, weights_path, conf_score=0.25):
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
        preds = self.model(frame, imgsz=640, conf = self.conf_score)[0]
        boxes = preds.boxes
        boxes_lp = boxes.xyxy.detach().cpu().numpy()
        
        print("BEFORE LP : ", boxes_lp)
        # Normalize the bounding boxes
        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        boxes_lp[:, 0::2] /= frame.shape[1]
        boxes_lp[:, 1::2] /= frame.shape[0]
        
        print("AFTER LP : ", boxes_lp)

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

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class OCR():
    def __init__(self, path2weights):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        # parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
        # parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
        # parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
        # parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        opt = parser.parse_args()
        opt.Transformation = 'TPS'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'Attn'
        opt.saved_model = path2weights
        
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)
        opt.num_class = 38

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        self.model = torch.nn.DataParallel(model).to(device)
        state_dict = torch.load(opt.saved_model, map_location=device)
        self.model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        self.model.eval()
        self.model.to(device)
        self.transform = ResizeNormalize((100, 32))
        self.opt = opt


    def __call__(self, img):
        
        image_tensors = [self.transform(image) for image in [img]] * 10
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        # print(image_tensors.shape)
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)
        if 'CTC' in self.opt.Prediction:
            preds = self.model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index, preds_size)

        else:
            preds = self.model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        
        return str(pred), float(confidence_score.item())

class LicensePlateModel():
    def __init__(self, det_weights, rec_weights, det_only=False):
        self.detector = LicensePlateDet(det_weights)
        if(not det_only):
            self.recognizor = OCR(rec_weights)
        else:
            self.recognizor = None
    
    def __call__(self, frame):
        # Detect First
        bboxes, scores, _ = self.detector(frame)
        # Recognize the detections
        lp_numbers = []
        lp_confs = []
        for box in bboxes:
            l, t, r, b = box.astype(np.int32)
            print("License Plate Box:", l, t, r, b)
            crop = frame[t:b, l:r, :]
            crop = deskew_plate(crop)
            if(crop is None):
                lp_numbers.append(None)
                lp_confs.append(0)
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            lp_box_img = Image.fromarray(crop).convert('L')
            if(self.recognizor is not None):
                lp_num, current_conf = self.recognizor(lp_box_img)
                print(lp_num, current_conf)
            else:
                lp_num, current_conf = None, None
            lp_numbers.append(lp_num)
            lp_confs.append(current_conf)
        
        return bboxes, lp_numbers, lp_confs

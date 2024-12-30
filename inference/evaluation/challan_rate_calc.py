import os
import sys
from ultralytics import YOLO
from instance_funcs import *
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import torchvision.transforms as transforms
from PIL import Image
import editdistance
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# sys.path.insert(0, '/home2/deepti.rawat/home/keshav/deep-text-recognition-benchmark/modules')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append("ultralytics/")

challan_rate_dir = '../hnh_instance_inference/challan_rate_calc_input'
gt_folder = '../Video_set_1/gt_annots'
videos_folder = '../Video_set_1/videos'
lp_det_model = YOLO('../license_plate_detection_train/weights/best.pt')
folders = os.listdir(challan_rate_dir)

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


def format_boxes_new_xyxy(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[1] * image_height)
        xmin = int(box[0] * image_width)
        ymax = int(box[3] * image_height)
        xmax = int(box[2] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

def get_lp_box(roi_frame):
    preds = lp_det_model(roi_frame, classes = [0,1], conf=0.4)[0]

        
    # Get the detections
    lp_boxes, lp_scores, lp_classes = getDetections(preds, roi_frame)
    
    # Bounding boxes are in normalized ymin, xmin, ymax, xmax
    original_h, original_w, _ = roi_frame.shape

    # The tracker will accept boxes in the format (xc, yc, w, h)
    lp_boxes = format_boxes_new_xyxy(lp_boxes, original_h, original_w)

    # return the lp_box with the largest score
    if len(lp_boxes) > 0:
        max_score_index = np.argmax(lp_scores)
        return lp_boxes[max_score_index]
    else:
        return None

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
        # self.AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.model.eval()
        self.model.to(device)
        self.transform = ResizeNormalize((100, 32))
        self.opt = opt


    def infer(self, img):
        
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


ocr = OCR('/ssd_scratch/cvit/keshav/dashcop/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth')

def calculate_cer(gt, pred):
    # Calculate the edit distance
    edit_dist = editdistance.eval(gt, pred)
    # CER is the edit distance divided by the length of the ground truth
    cer = edit_dist / len(gt) if len(gt) > 0 else 0
    return cer



def correct_and_remove_spaces(og_text):
    if og_text == None:
        return None
    text = og_text
    # only keep alphabets and numbers in text
    text = ''.join(e for e in text if e.isalnum())
    # convert to upper case
    text = text.upper()
    gt_corrections_txt_file = 'gt_lp_correction.txt'
    found = False
    new_text = None
    with open(gt_corrections_txt_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            if 'lp_box' in line:
               words = line.split('_')
               if words[-2] == text:
                    new_text = words[-1].split('.')[0]
                    new_text = ''.join(e for e in new_text if e.isalnum())
                    # convert to upper case
                    new_text = new_text.upper()
                    print(f"Corrected {text} to {new_text}")
                    found = True
            else:
                if text == line:
                    print(f"Removed {text}")
                    return None
    if found:
        return new_text
    return og_text

def validate_lp(gt_lp):
    state_codes = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DD', 'DL', 'LD', 'PY']
    gt = False
    gt_lp = gt_lp.upper()
    gt_lp = ''.join(e for e in gt_lp if e.isalnum())
    if gt_lp == None or len(gt_lp) <= 8 :
        return False

    
    for state in state_codes:
        if state in gt_lp:
            gt = True
    if gt:
        return True
    else:
        return False
        
    return False

conf_mat_challan_rate_by_lp_box = np.zeros((3,3))
conf_mat_challan_rate_by_lp_num = np.zeros((3,3))
dictionary_gt = {}
avg_cer = 0
total_num = 0

for folder in folders:

    # skip if folder is empty, then skip
    if len(os.listdir(os.path.join(challan_rate_dir, folder))) == 0:
        continue
    video_name = folder + '.mp4'
    # if '50729' not in video_name:
    #     continue
    all_frames = []
    cap = cv2.VideoCapture(os.path.join(videos_folder, video_name))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    gt_annotation_file = os.path.join(gt_folder, folder, 'annotations.xml')
    gt_tree = ET.parse(gt_annotation_file)
    gt_root = gt_tree.getroot()
    
    matched_ids_file = os.path.join(challan_rate_dir, folder, 'matched_ids.txt')
    false_negative_file = os.path.join(challan_rate_dir, folder, 'false_negative_background_tracks.txt')

    gt_rider_id_to_assoc_id = {}
    for track in gt_tree.iter('track'):
        if track.attrib['label'] == 'rider':
                # loop over all box tags
            for box in track.iter('box'):
                # get the attributes of the box tag
                if box.attrib['outside'] == '0' and box.attrib['occluded'] == '0':
                    # get attribute tag inside box tag
                    for attribute in box.iter('attribute'):
                        if attribute.attrib['name'] == 'association_id':
                            gt_rider_id_to_assoc_id[track.attrib['id']] = attribute.text

    gt_motorcycle_id_to_assoc_id = {}
    gt_motorcycle_id_to_boxes = {}
    for track in gt_tree.iter('track'):
        if track.attrib['label'] == 'motorcycle':
            # loop over all box tags
            for box in track.iter('box'):
                # get the attributes of the box tag
                if box.attrib['outside'] == '0' and box.attrib['occluded'] == '0':
                    # get attribute tag inside box tag
                    if track.attrib['id'] not in gt_motorcycle_id_to_boxes:
                        gt_motorcycle_id_to_boxes[track.attrib['id']] = []
                    gt_motorcycle_id_to_boxes[track.attrib['id']].append([box.attrib['frame'], box.attrib['xtl'], box.attrib['ytl'], box.attrib['xbr'], box.attrib['ybr']])
                    for attribute in box.iter('attribute'):
                        if attribute.attrib['name'] == 'motor_track_id':
                            gt_motorcycle_id_to_assoc_id[track.attrib['id']] = attribute.text

    gt_rider_id_to_boxes = {}
    for track in gt_tree.iter('track'):
        if track.attrib['label'] == 'rider':
            # loop over all box tags
            for box in track.iter('box'):
                # get the attributes of the box tag
                if box.attrib['outside'] == '0' and box.attrib['occluded'] == '0':
                    # get attribute tag inside box tag
                    if track.attrib['id'] not in gt_rider_id_to_boxes:
                        gt_rider_id_to_boxes[track.attrib['id']] = []
                    gt_rider_id_to_boxes[track.attrib['id']].append([box.attrib['frame'], box.attrib['xtl'], box.attrib['ytl'], box.attrib['xbr'], box.attrib['ybr']])

    gt_assoc_id_to_motorcycle_id = {}
    # invert the dictionary
    for key, value in gt_motorcycle_id_to_assoc_id.items():
        gt_assoc_id_to_motorcycle_id[value] = key
    
    gt_motorcycle_id_to_lp_number = {}
    for motor_id in gt_motorcycle_id_to_assoc_id.keys():
        best_num_char = None
        for track in gt_tree.iter('track'):
            if track.attrib['label'] == 'license_plate':
                for box in track.iter('box'):
                    if box.attrib['outside'] == '0' and box.attrib['occluded'] == '0':
                        curr_motor_box = None
                        for motor_box in gt_motorcycle_id_to_boxes[motor_id]:
                            if int(motor_box[0]) == int(box.attrib['frame']):
                                curr_motor_box = motor_box
                        
                        if curr_motor_box == None:
                            continue

                        # check if license plate box is inside motorcycle box
                        if float(curr_motor_box[1]) <= float(box.attrib['xtl']) and float(curr_motor_box[2]) <= float(box.attrib['ytl']) and float(curr_motor_box[3]) >= float(box.attrib['xbr']) and float(curr_motor_box[4]) >= float(box.attrib['ybr']):
                            lp_text = ''
                            for attribute in box.iter('attribute'):
                                if attribute.attrib['name'] == 'lp_number':
                                    lp_text = attribute.text
                            if lp_text == None:
                                continue
                            # only keep alphabets and numbers in lp_text
                            lp_text = ''.join(e for e in lp_text if e.isalnum())
                            if (best_num_char == None or len(lp_text) > best_num_char) and len(lp_text) > 8:
                                best_num_char = len(lp_text)
                                gt_motorcycle_id_to_lp_number[motor_id] = lp_text


    print(gt_rider_id_to_assoc_id)
    print(gt_motorcycle_id_to_assoc_id)
    print(gt_assoc_id_to_motorcycle_id)
    print("LICENSE PLATES DICT")
    print(gt_motorcycle_id_to_lp_number)
    print(len(gt_motorcycle_id_to_assoc_id.keys()))
    print(len(gt_motorcycle_id_to_lp_number.keys()))


    with open(matched_ids_file, 'r') as f:
        matched_ids = f.readlines()
        matched_ids = [x.strip() for x in matched_ids]
        for line in matched_ids:
            data = line.split(',')
            gt_rider_id = data[0]
            gt_rider_label = data[1]
            pred_rider_label = data[2]
            pred_rider_ids = data[3:]
            # print(gt_rider_id, gt_rider_label, pred_rider_label, pred_rider_ids)

            all_roi_crops_to_lp_boxes = {}
            gt_lp_num = None
            pred_lp_num = None
            max_conf = None
            pred_selected_lp_img = None
            predicted_double_line_plate = None
            predicted_roi_frame = None
            predicted_frame_num = None

            if gt_rider_id in gt_rider_id_to_assoc_id:
                gt_assoc_id = gt_rider_id_to_assoc_id[gt_rider_id]
                if gt_assoc_id in gt_assoc_id_to_motorcycle_id:
                    gt_motorcycle_id = gt_assoc_id_to_motorcycle_id[gt_assoc_id]
                    if gt_motorcycle_id in gt_motorcycle_id_to_lp_number:
                        gt_lp_number = gt_motorcycle_id_to_lp_number[gt_motorcycle_id]
                        gt_lp_num = correct_and_remove_spaces(gt_lp_number)
                     
                        if gt_lp_num == None:
                            continue
                        print("hello")
                        print(gt_rider_id, gt_rider_label, pred_rider_label, pred_rider_ids, gt_lp_num)

                        # find rider ids which have the assoc_id same as gt_assoc_id
                        rider_ids = []
                        for rider_id, assoc_id in gt_rider_id_to_assoc_id.items():
                            if assoc_id == gt_assoc_id:
                                rider_ids.append(rider_id)
                        # print(rider_ids)

                        for motor_box in gt_motorcycle_id_to_boxes[gt_motorcycle_id]:
                            frame_num = motor_box[0]
                            frame = all_frames[int(frame_num)]
                            motor_xmin, motor_ymin, motor_xmax, motor_ymax = int(float(motor_box[1])), int(float(motor_box[2])), int(float(motor_box[3])), int(float(motor_box[4]))
                            rider_boxes = []
                            for rider_id in rider_ids:
                                for rider_box in gt_rider_id_to_boxes[rider_id]:
                                    if rider_box[0] == frame_num:
                                        rider_xmin, rider_ymin, rider_xmax, rider_ymax = int(float(rider_box[1])), int(float(rider_box[2])), int(float(rider_box[3])), int(float(rider_box[4]))
                                        rider_boxes.append([rider_xmin, rider_ymin, rider_xmax, rider_ymax])
                            if len(rider_boxes) == 0:
                                continue
                            
                            # get the roi_xmin which is the minimum of all the xmins of the riders on the motorcycle and the xmin of the motorcycle
                            roi_xmin = min(motor_xmin, min([rider[0] for rider in rider_boxes]))
                            # get the roi_ymin which is the minimum of all the ymins of the riders on the motorcycle and the ymin of the motorcycle
                            roi_ymin = min(motor_ymin, min([rider[1] for rider in rider_boxes]))
                            # get the roi_xmax which is the maximum of all the xmaxs of the riders on the motorcycle and the xmax of the motorcycle
                            roi_xmax = max(motor_xmax, max([rider[2] for rider in rider_boxes]))
                            # get the roi_ymax which is the maximum of all the ymaxs of the riders on the motorcycle and the ymax of the motorcycle
                            roi_ymax = max(motor_ymax, max([rider[3] for rider in rider_boxes]))

                            # get the roi_width and roi_height
                            roi_width = roi_xmax - roi_xmin
                            roi_height = roi_ymax - roi_ymin
                            roi_frame = frame[int(roi_ymin):int(roi_ymax), int(roi_xmin):int(roi_xmax)]

                            lp_box = get_lp_box(roi_frame)
                            all_roi_crops_to_lp_boxes[frame_num] = lp_box
                            if lp_box is not None and len(gt_lp_num) > 8:
                                lp_box_img = roi_frame[int(lp_box[1]):int(lp_box[3]), int(lp_box[0]):int(lp_box[2])]
                                double_line = lp_box_img
                                lp_box_img = deskew_plate(lp_box_img)
                                # current_conf = roi_width * roi_height
                                if lp_box_img is None:
                                    continue
                                # convert to RGB from BGR
                                lp_box_img = cv2.cvtColor(lp_box_img, cv2.COLOR_BGR2RGB)
                                # convert to PIL image
                                lp_box_img = Image.fromarray(lp_box_img).convert('L')
                                lp_text_pred, current_conf = ocr.infer(lp_box_img)
                                if len(lp_text_pred) > 0:
                                    lp_text_pred = ''.join(e for e in lp_text_pred if e.isalnum())
                                if len(lp_text_pred) > 8:
                                    if pred_lp_num == None or current_conf > max_conf:
                                        pred_lp_num = lp_text_pred
                                        max_conf = current_conf
                                        # convert to cv2 from PIL
                                        lp_box_img = np.array(lp_box_img)
                                        pred_selected_lp_img = lp_box_img
                                        predicted_double_line_plate = double_line
                                        predicted_roi_frame = roi_frame
                                        predicted_frame_num = frame_num


            lp_box = None
            if gt_lp_num != None and validate_lp(gt_lp_num):     
                for roi_crop in all_roi_crops_to_lp_boxes.keys():
                    lp_box = all_roi_crops_to_lp_boxes[roi_crop]
                    if lp_box is None:
                        continue
                    # true positive helmet class
                    if gt_rider_label == pred_rider_label and pred_rider_label == '0':
                        conf_mat_challan_rate_by_lp_box[0][0] += 1
                    # true positive no helmet class
                    elif gt_rider_label == pred_rider_label and pred_rider_label == '1':
                        conf_mat_challan_rate_by_lp_box[1][1] += 1
                    # misclassification of helmet as no helmet
                    elif gt_rider_label == '0' and pred_rider_label == '1':
                        conf_mat_challan_rate_by_lp_box[1][0] += 1

                    # misclassification of no helmet as helmet
                    elif gt_rider_label == '1' and pred_rider_label == '0':
                        conf_mat_challan_rate_by_lp_box[0][1] += 1
                    break

                if lp_box is None:
                    if gt_rider_label == '0':
                        conf_mat_challan_rate_by_lp_box[2][0] += 1
                        conf_mat_challan_rate_by_lp_num[2][0] += 1
                    elif gt_rider_label == '1':
                        conf_mat_challan_rate_by_lp_box[2][1] += 1
                        conf_mat_challan_rate_by_lp_num[2][1] += 1
            
                gt_lp_num = correct_and_remove_spaces(gt_lp_num)
                if gt_lp_num == None:
                    continue
  
                # convert to small case

                
                # print(gt_lp_num, pred_lp_num)
                # if pred_lp_num == 'TS08HU8941':
                #     print("FOUND")
                #     print(gt_lp_num, pred_lp_num)
                #     print(folder)
                #     exit()
                # else:
                #     continue
                if pred_lp_num != None and validate_lp(pred_lp_num):
                    gt_lp_num = gt_lp_num.upper()
                    pred_lp_num = pred_lp_num.upper()
                    print("hi")
                    print("INFERRREDDDD")
                    print(gt_lp_num)
                    print(pred_lp_num, current_conf)
                    cer = calculate_cer(gt_lp_num, pred_lp_num)
                    avg_cer += cer
                    if gt_lp_num == pred_lp_num and pred_rider_label == gt_rider_label and pred_rider_label == '0':
                        conf_mat_challan_rate_by_lp_num[0][0] += 1
                    elif gt_lp_num == pred_lp_num and pred_rider_label == gt_rider_label and pred_rider_label == '1':
                        conf_mat_challan_rate_by_lp_num[1][1] += 1
                        print("TRUE POSITIVE")
                        print(gt_lp_num, predicted_frame_num)
                        if dictionary_gt.get(gt_lp_num) == None:
                            dictionary_gt[gt_lp_num] = 1
                        else:
                            dictionary_gt[gt_lp_num] += 1
                        cv2.imwrite('true_positives/lp_box' + str(rider_id) +'_' +str(gt_lp_num) + '_' + str(pred_lp_num) + '.jpg', pred_selected_lp_img)
                        cv2.imwrite('true_positives/double_line' + str(rider_id)+ '_' +str(gt_lp_num) + '_' + str(pred_lp_num) + '.jpg', predicted_double_line_plate)
                        cv2.imwrite('true_positives/roi_frame' + str(rider_id)+ '_' +str(gt_lp_num) + '_' + str(pred_lp_num) + '.jpg', predicted_roi_frame)
                    elif gt_lp_num != pred_lp_num and pred_rider_label == '0':
                        conf_mat_challan_rate_by_lp_num[0][2] += 1
                    elif gt_lp_num != pred_lp_num and pred_rider_label == '1':
                        conf_mat_challan_rate_by_lp_num[1][2] += 1
                        print("FALSE POSITIVES")
                        if dictionary_gt.get(gt_lp_num) == None:
                            dictionary_gt[gt_lp_num] = 1
                        else:
                            dictionary_gt[gt_lp_num] += 1
                        print(gt_lp_num, predicted_frame_num)
                        cv2.imwrite('temp_plates/lp_box' + str(rider_id)+ '_' + str(gt_lp_num) + '_' + str(pred_lp_num) + '.jpg', pred_selected_lp_img)
                        cv2.imwrite('temp_plates/double_line' + str(rider_id)+'_' + str(gt_lp_num) + '_' + str(pred_lp_num) + '.jpg', predicted_double_line_plate)
                        cv2.imwrite('temp_plates/roi_frame' + str(rider_id)+ '_' +str(gt_lp_num) + '_' + str(pred_lp_num) + '.jpg', predicted_roi_frame)

            if gt_lp_num != None and pred_lp_num != None and len(gt_lp_num) > 8 and len(pred_lp_num) > 8:
                cer = calculate_cer(gt_lp_num, pred_lp_num)
                avg_cer += cer
                total_num += 1
    state_codes = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DD', 'DL', 'LD', 'PY']
    with open(false_negative_file, 'r') as f:
        matched_ids = f.readlines()
        matched_ids = [x.strip() for x in matched_ids]
        for line in matched_ids:
            data = line.split(',')
            gt_rider_id = data[0]
            gt_rider_label = data[1]
            # print(gt_rider_id, gt_rider_label, pred_rider_label, pred_rider_ids)

            all_roi_crops_to_lp_boxes = {}

            if gt_rider_id in gt_rider_id_to_assoc_id:
                gt_assoc_id = gt_rider_id_to_assoc_id[gt_rider_id]
                if gt_assoc_id in gt_assoc_id_to_motorcycle_id:
                    gt_motorcycle_id = gt_assoc_id_to_motorcycle_id[gt_assoc_id]
                    if gt_motorcycle_id in gt_motorcycle_id_to_lp_number:
                        gt_lp_number = gt_motorcycle_id_to_lp_number[gt_motorcycle_id]
                        gt_lp_number = gt_lp_number.upper()
                        gt_lp_number = ''.join(e for e in gt_lp_number if e.isalnum())
                        if gt_lp_number == None or len(gt_lp_number) <= 8 :
                            continue
                        is_state_code = False
                        for state in state_codes:
                            if state in gt_lp_number:
                                is_state_code = True
                        if not is_state_code:
                            continue

                        #updtae conf mat for false negatives 
                        if gt_rider_label == '0':
                            conf_mat_challan_rate_by_lp_box[2][0] += 1
                            conf_mat_challan_rate_by_lp_num[2][0] += 1
                        elif gt_rider_label == '1':
                            conf_mat_challan_rate_by_lp_box[2][1] += 1
                            conf_mat_challan_rate_by_lp_num[2][1] += 1
                            


print("Confusion matrix for challan rate by lp box")
print(conf_mat_challan_rate_by_lp_box)
print("Confusion matrix for challan rate by lp number")
print(conf_mat_challan_rate_by_lp_num)

print("CER")
print(avg_cer)

print("Avg cer")
print(avg_cer/total_num)

print("Dictionary")
print(dictionary_gt)

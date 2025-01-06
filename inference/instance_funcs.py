# from core.association import *
import pandas as pd
import numpy as np
import cv2
import sys
from easydict import EasyDict as edict
import torch
import yaml
import os
from PIL import Image

sys.path.append('../')

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

def predictDepth(image, image_processor, depth_model):
    '''
    image is a np array of shape (3, H, W)
    image_processor is the image pre-processor for the depth prediction model
    depth_model is the depth prediction model
    predicts the depth of the image, and returns an np array of shape (H, W)
    '''
    image = Image.fromarray(image)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()

    # For Visualization of Depth
    formatted = (output * 255 / np.max(output)).astype("uint8")

    return output, formatted

def getDetections(preds, frame):
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

def calculate_iou(box1, box2):
    # Extract coordinates for the first bounding box
    x1, y1, x2, y2 = box1
    # Extract coordinates for the second bounding box
    x3, y3, x4, y4 = box2

    # Calculate area of intersection
    intersection_width = max(0, min(x2, x4) - max(x1, x3))
    intersection_height = max(0, min(y2, y4) - max(y1, y3))
    area_intersection = intersection_width * intersection_height

    # Calculate areas of individual bounding boxes
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)

    # Calculate area of union
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate IoU
    iou = area_intersection / max(area_union, 1e-10)  # Avoid division by zero

    return iou

def calculate_intersection(box1,box2):

    # Extract coordinates for the first bounding box
    x1, y1, x2, y2 = box1
    # Extract coordinates for the second bounding box
    x3, y3, x4, y4 = box2

    # Calculate area of intersection
    intersection_width = max(0, min(x2, x4) - max(x1, x3))
    intersection_height = max(0, min(y2, y4) - max(y1, y3))
    area_intersection = intersection_width * intersection_height

    return area_intersection

def getDepthMasks(preds, depth_mask):
    masks = []
    depths = []

    for i in range(len(preds.masks.data)):
        int_mask = preds.masks.data[i].cpu().numpy().astype(np.int32)
        if(depth_mask is not None):
            obj_mask = int_mask * depth_mask
            if(len(int_mask[int_mask != 0]) != 0):
                avg_depth = obj_mask.sum() / len(int_mask[int_mask != 0])
            else:
                avg_depth = 0
            depths.append(avg_depth)
        masks.append(int_mask)
    
    depths = np.array(depths)
    masks = np.array(masks)

    return depths, masks

def extract_roi(frame, rider, motorcycle):
    """
    args:
    frame : np.array
    rider, motorcycle : pd.DataFrame

    output:
    roi_instances : list of np.array
    """
    roi_instances = []
    for i in range(len(motorcycle)):
        motor = motorcycle.loc[motorcycle['instance_id'] == i]
        instance = motorcycle.iloc[i]['instance_id']
        ride = rider.loc[rider['instance_id'] == instance]

        if (len(ride) == 0):
            continue

        xmax = max(float(motor['x'] + motor['w']/2),
                   max(ride['x'] + ride['w']/2))
        xmin = min(float(motor['x'] - motor['w']/2),
                   min(ride['x'] - ride['w']/2))
        ymax = max(float(motor['y'] + motor['h']/2),
                   max(ride['y'] + ride['h']/2))
        ymin = min(float(motor['y'] - motor['h']/2),
                   min(ride['y'] - ride['h']/2))

        w = xmax - xmin
        h = ymax - ymin

        xmax = xmax + 0.05*w
        xmin = xmin - 0.05*w

        ymax = ymax + 0.05 * h
        ymin = ymin - 0.05 * h

        if (xmin < 0):
            xmin = 0
        if (xmax > 1):
            xmax = 1
        if (ymax > 1):
            ymax = 1
        if(ymin < 0):
            ymin = 0

        t = int(ymin*frame.shape[0])
        l = int(xmin*frame.shape[1])
        b = int(ymax*frame.shape[0])
        r = int(xmax*frame.shape[1])

        if t < 0 or l < 0 or b < 0 or r < 0:
            continue

        roi_frame = frame[t:b, l:r]
        # roi_frame = frame
        original_position = (t, l, b, r)
        roi_dict = {'frame': roi_frame,
                    'original_position': original_position, 'num_riders': len(ride)}
        roi_instances.append(roi_dict)

    return roi_instances


def make_roi(motorcycle, riders):
    # print(motorcycle)
    # print(riders)
    all = np.concatenate(
        [np.array(riders), np.array(motorcycle)[None]], axis=0)
    print(all)
    t = int(np.min(all[:, 1]))
    l = int(np.min(all[:, 0]))
    b = int(np.max(all[:, 3]))
    r = int(np.max(all[:, 2]))
    print(t, l, b, r)

    return np.array([l, t, r, b])


# Given y : Trapezium coordinates (x,y, offsets ...) and a single rider (x,y,w,h), the function returns the IOU between the two
def trapez_rider_iou(y_, rider):

    y_ = [[y_[0], y_[1]], [y_[2], y_[3]], [y_[4], y_[5]], [y_[6], y_[7]]]
    x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
    rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
    print("####")
    print(y_)
    print(rider)
    print("####")
    return iou(y_, rider)

# Given the trapeziums and riders of a frame, the function compares IOU between all trapeziums and riders
# and assigns instance IDs to the riders based on the IOU threshold


def get_instance_with_trapez(rider, trapezium, iou_threshold):
    """
    args:
    rider, trapezium : pd.DataFrame

    output:
    rider, trapezium : pd.DataFrame with a column named 'instance_id'
    """
    # print info of the rider and trapezium dataframes
    # print("Rider dataframe info:")
    # print(rider.info())
    trapezium_instance_ids = np.zeros(len(trapezium))
    for i in range(len(trapezium)):
        trapezium_instance_ids[i] = i
        for j in range(len(rider)):
            if (trapez_rider_iou(trapezium[i], rider.iloc[j]) > iou_threshold):
                if (rider.iloc[j]['instance_id'] == -1):
                    rider.iat[j, rider.columns.get_loc('instance_id')] = i
                else:
                    instance = int(rider.iloc[j]['instance_id'])
                    if (trapez_rider_iou(trapezium[instance], rider.iloc[j]) < trapez_rider_iou(trapezium[i], rider.iloc[j])):
                        rider.iat[j, rider.columns.get_loc('instance_id')] = i
                    else:
                        rider.iat[j, rider.columns.get_loc(
                            'instance_id')] = instance
    return rider, trapezium_instance_ids

# This function is not used as of now. If function call is uncommented in main, it will basically annotate the frame with the boxes and labels and store it in the output folder


def save_annotated_frame(frame, bboxes, classes, scores, num_objects, frame_num):
    """
    args:
    frame : np.array
    bboxes : np.array
    classes : np.array
    scores : np.array
    num_objects : int
    frame_num : int

    output:
    None
    """
    # save annotated frame

    output_path = 'outputs/frames'
    colors = [(0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    for i in range(num_objects):
        # save bbox
        # convert bbox from yolo format to opencv format
        current_class = int(classes[i])
        bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if current_class == 1 or current_class == 2:
            bbox = [bboxes[i][1] - bboxes[i][3] / 2, bboxes[i][0] - bboxes[i][2] /
                    2, bboxes[i][1] + bboxes[i][3] / 2, bboxes[i][0] + bboxes[i][2] / 2]
        bbox = [int(bbox[0] * frame.shape[0]), int(bbox[1] * frame.shape[1]),
                int(bbox[2] * frame.shape[0]), int(bbox[3] * frame.shape[1])]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]), colors[current_class], 2)
        # save class
        cv2.putText(frame, str(current_class), (int(bboxes[i][1]), int(
            bboxes[i][0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # save score
        cv2.putText(frame, str(scores[i]), (int(bboxes[i][1]), int(
            bboxes[i][2])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(output_path + '/frame_' + str(frame_num) + '.jpg', frame)
    # save file
    output_path = 'outputs/annotations'
    with open(output_path + '/frame_' + str(frame_num) + '.txt', 'w') as f:
        for i in range(num_objects):
            # convert bbox from yolo format to opencv format
            bbox = [bboxes[i][1] - bboxes[i][3] / 2, bboxes[i][0] - bboxes[i][2] /
                    2, bboxes[i][1] + bboxes[i][3] / 2, bboxes[i][0] + bboxes[i][2] / 2]
            bbox = [int(bbox[0] * frame.shape[0]), int(bbox[1] * frame.shape[1]),
                    int(bbox[2] * frame.shape[0]), int(bbox[3] * frame.shape[1])]
            # convert to int
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            f.write(str(int(classes[i])) + ' ' + str(scores[i]) + ' ' + str(
                bbox[1]) + ' ' + str(bbox[0]) + ' ' + str(bbox[3]) + ' ' + str(bbox[2]) + '\n')


# assigns a unique instance id to each motorcycle. Then loops over all riders and assigns the same instance id to the rider if the IOU is greater than the threshold.
# If the rider is already assigned to a motorcycle, then the iou of rider with the 2 motorcycles is calculated and the rider is assigned to the motorcycle with the higher iou.
# Updated rider and motorcycle dataframes are returned.
def get_instance(rider, motorcycle, iou_threshold):
    """
    args:
    rider, motorcycle : pd.DataFrame

    output:
    rider, motorycle : pd.DataFrame with a column named 'instance_id'
    """

    rider['instance_id'] = -1
    motorcycle['instance_id'] = -1

    for i in range(len(motorcycle)):
        motorcycle.iat[i, motorcycle.columns.get_loc('instance_id')] = i
        for j in range(len(rider)):
            if (motor_rider_iou(motorcycle.iloc[i], rider.iloc[j]) > iou_threshold):
                if (rider.iloc[j]['instance_id'] == -1):
                    rider.iat[j, rider.columns.get_loc('instance_id')] = i
                else:
                    instance = int(rider.iloc[j]['instance_id'])
                    instance_final = motor2_rider_iou(
                        motorcycle.iloc[i], motorcycle.iloc[instance], rider.iloc[j], i, instance)
                    rider.iat[j, rider.columns.get_loc(
                        'instance_id')] = instance_final

    return rider, motorcycle


def get_instance_tracks(tracklets_bbox, tracklets_cls, tracklets_ids, iou_threshold):

    riders = tracklets_bbox[tracklets_cls == 0]
    motorcycle = tracklets_bbox[tracklets_cls == 3]
    motor_tracklets_ids = tracklets_ids[tracklets_cls == 3]

    rider_id = np.ones(riders.shape[0], dtype=np.int32) * -1
    motor_id = np.ones(motorcycle.shape[0], dtype=np.int32) * -1
    motor_dict = {}
    motor_assocs = {}

    for i in range(motorcycle.shape[0]):
        motor_id[i] = i
        motor_dict[motor_tracklets_ids[i]] = i
        curr_riders = []
        for j in range(len(riders)):
            if (motor_rider_iou_tracks(motorcycle[i], riders[j]) > iou_threshold):
                if (rider_id[j] == -1):
                    rider_id[j] = i
                    curr_riders.append(riders[j])
                else:
                    instance = rider_id[j]
                    instance_final = motor2_rider_iou_tracks(
                        motorcycle[i], motorcycle[instance], riders[j], i, instance)
                    rider_id[j] = instance_final
                    curr_riders.append(riders[j])
        motor_assocs[motor_tracklets_ids[i]] = curr_riders

    return motor_assocs, motor_dict


# below functions are used as helper functions in the trapezium regressor. They handle corner cases for the predicted trapezium

def heuristic_on_pred(a, motor, rider_ins):
    no_of_bbox = len(motor) + len(rider_ins)
    if (motor['w'] == 0):
        no_of_bbox = no_of_bbox - 1

    mean_w = (rider_ins['w'].sum() + motor['w'].sum())/no_of_bbox
    mean_x = (rider_ins['x'].sum() + motor['x'].sum())/no_of_bbox

    if (a[4] < mean_w):
        a[4] = motor['w'].mean()
    if (a[0] < mean_x - mean_w/2):
        a[0] = rider_ins['x'].mean()
    if (a[0] > mean_x + mean_w/2):
        a[0] = rider_ins['x'].mean()
    return a


def corner_condition(y, xmax, ymax):
    if (y[0] < 0):
        y[0] = 0
    if (y[0] > xmax):
        y[0] = xmax
    if (y[1] < 0):
        y[1] = 0
    if (y[1] > ymax):
        y[1] = ymax
    if (y[2] < 0):
        y[2] = 0
    if (y[2] > xmax):
        y[2] = xmax
    if (y[3] < 0):
        y[3] = 0
    if (y[3] > ymax):
        y[3] = ymax
    if (y[4] < 0):
        y[4] = 0
    if (y[4] > xmax):
        y[4] = xmax
    if (y[5] < 0):
        y[5] = 0
    if (y[5] > ymax):
        y[5] = ymax
    if (y[6] < 0):
        y[6] = 0
    if (y[6] > xmax):
        y[6] = xmax
    if (y[7] < 0):
        y[7] = 0
    if (y[7] > ymax):
        y[7] = ymax

    return y

# This function basically takes a bbox and matches the motorcycle with the bbox with the least distance. It returns the instance id of that motorcycle and the number of riders on that motorcycle. Used in tracking


def find(bbox, instance, bboxes, classes):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    flag = 10000
    idx = -1
    for i in range(len(bboxes)):
        if (classes[i] == 'Motorcycle'):
            b = bboxes[i]
            val = abs(xmin-b[0]) + abs(ymin-b[1]) + \
                abs(xmax-b[0]-b[2]) + abs(ymax-b[1]-b[3])
            if (val < flag):
                flag = val
                idx = i
    if (idx != -1):
        num, num_riders = instance[idx][0], instance[idx][1]
    else:
        num, num_riders, val = -1, -1, flag
    return num, num_riders, val


def find_(bbox, instance, bboxes, classes):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    flag = 10000
    idx = -1
    for i in range(len(bboxes)):
        if (classes[i] == 'Motorcycle'):
            b = bboxes[i]
            print(bbox)
            print(b)
            val = abs(xmin-b[0]) + abs(ymin-b[1]) + \
                abs(xmax-b[2]) + abs(ymax-b[3])
            if (val < flag):
                flag = val
                idx = i
    if (idx != -1):
        num, num_riders = instance[idx][0], instance[idx][1]
    else:
        num, num_riders, val = -1, -1, flag
    return num, num_riders, val


def get_assosciation(motor, rider, motor_mask, rider_mask, printF):
    '''
    motor is the bounding box corresponding to the motorcycle
    '''
    # print(motor)
    # print(rider)
    # x,y,w,h = motor[0], motor[1], motor[2], motor[3]
    # motor = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
    # x,y,w,h = rider[0], rider[1], rider[2], rider[3]
    # rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
    # return iou(motor, rider)
    # count number of 1s in motor_mask and rider_mask
    rider_mask_size = len(rider_mask[rider_mask != 0])
    motor_mask_size = len(motor_mask[motor_mask != 0])
    # print("R: ", rider_mask_size, "M: ", motor_mask_size)
    kernel = np.ones([5, 5], dtype=np.uint8)
    motor_mask_dil = cv2.dilate(motor_mask, kernel, iterations=2)
    rider_mask_dil = cv2.dilate(rider_mask, kernel, iterations=2)
    intersection = motor_mask_dil & rider_mask_dil
    num_common = len(intersection[intersection != 0])
    if printF:
        print(num_common)
    rider_mask_size = len(rider_mask_dil[rider_mask_dil != 0])
    motor_mask_size = len(motor_mask_dil[motor_mask_dil != 0])
    # print("After dilation")
    # print("R: ", rider_mask_size, "M: ", motor_mask_size)
    union = motor_mask_dil | rider_mask_dil
    num_union = len(union[union != 0])
    if(printF):
        print(num_union)
    # print("Intersection: ", num_common, "Union: ", num_union)
    return num_common / num_union


# Step 1: Apply Canny edge detection to the binary masks.
def apply_canny(mask):
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 1,
                      1)  # Adjust parameters as needed
    return edges

# Step 2: Convert the edge images to lists of coordinates for edge points.


def image_to_edge_points(image):
    edge_points = np.column_stack(np.where(image == 255))
    return edge_points

from scipy.spatial import distance

def get_assosciation2(motor, rider, motor_mask, rider_mask):
    '''
    motor is the bounding box corresponding to the motorcycle
    '''

    kernel = np.ones([5, 5], dtype=np.uint8)
    motor_mask_dil = cv2.dilate(motor_mask, kernel, iterations=2)
    rider_mask_dil = cv2.dilate(rider_mask, kernel, iterations=2)
    intersection = motor_mask_dil & rider_mask_dil
    num_common = len(intersection[intersection != 0])

    union = motor_mask_dil | rider_mask_dil
    num_union = len(union[union != 0])

    edges_rider = apply_canny(motor_mask_dil)
    edges_motor = apply_canny(rider_mask_dil)

    edge_points_motor = image_to_edge_points(edges_motor)
    edge_points_rider = image_to_edge_points(edges_rider)
    # Step 3: Calculate pairwise distances between all edge points.
    distances = distance.cdist(edge_points_motor, edge_points_rider)
    # Step 4: Compute the average distance.
    average_distance = np.mean(distances)
    average_distance = average_distance / (distances.shape[0] * distances.shape[1])

    return num_common / num_union, average_distance


def get_interpolated_bbox(current_rider_id, frame_num, all_tracks_data):
    # interpolate bbox for current rider in the current frame_num based on all frames before and after in all_tracks_data
    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    is_valid = False

    # loop across the dictionary all_tracks_data
    for i in all_tracks_data.keys():
        track_data = all_tracks_data[i]
        # print(track_data)
        bboxes = track_data[0]
        ids = track_data[1]
        clss = track_data[2]

        if current_rider_id in ids and i in range(frame_num - 7, frame_num + 7):
            # get the bbox for the current rider
            idx = np.where(ids == current_rider_id)[0][0]
            bbox = bboxes[idx]
            if clss[idx] == 0:
                x.append(i)
                y1.append(bbox[0])
                y2.append(bbox[1])
                y3.append(bbox[2])
                y4.append(bbox[3])
    

    if len(x) > 0 and frame_num >= x[0] and frame_num <= x[-1]:
        is_valid = True

    if not is_valid:
        return None
    
    # reset x to start from 0
    if frame_num < x[0] :
        x = [i - frame_num for i in x]
        frame_num = 0
    else:
        x = [i - x[0] for i in x]
        frame_num = frame_num - x[0]

    # convert to numpy arrays
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)

    print("Rider ID ", current_rider_id)
    for i in range(len(x)):
        print(x[i], y1[i], y2[i], y3[i], y4[i])

    # from a curve fit, get the bbox for the current frame_num
    degree = 2
    p1 = np.polyfit(x, y1, degree)
    p2 = np.polyfit(x, y2, degree)
    p3 = np.polyfit(x, y3, degree)
    p4 = np.polyfit(x, y4, degree)

    y1_new = np.polyval(p1, frame_num)
    y2_new = np.polyval(p2, frame_num)
    y3_new = np.polyval(p3, frame_num)
    y4_new = np.polyval(p4, frame_num)

    bbox = [y1_new, y2_new, y3_new, y4_new]

    return bbox

def transpose_roi_frame(boxes, roi_cords):
    height = roi_cords[3] - roi_cords[1]
    width = roi_cords[2] - roi_cords[0]
    
    for i, box in enumerate(boxes):
        # convert box from yolo format to xmin, ymin, xmax, ymax
        xmin = box[0] * width + roi_cords[0]
        ymin = box[1] * height + roi_cords[1]
        xmax = box[2] * width + roi_cords[0]
        ymax = box[3] * height + roi_cords[1]
        
        boxes[i] = [xmin, ymin, xmax, ymax]
        
    return boxes


def compute_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def remove_duplicate_masks(masks):
    indices_to_remove = set()
    n_masks = masks.shape[0]

    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            iou = compute_mask_iou(masks[i], masks[j])
            if iou > 0.7:
                if np.all(masks[i] == masks[j]):
                    indices_to_remove.add(j)
                elif np.all(masks[i] <= masks[j]):
                    indices_to_remove.add(i)
                else:
                    indices_to_remove.add(j)

    masks_filtered = np.delete(masks, list(indices_to_remove), axis=0)
    return masks_filtered, list(indices_to_remove)
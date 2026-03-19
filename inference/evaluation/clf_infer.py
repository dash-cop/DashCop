import os
import sys


# Add parent directory and pipeline_comp to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # inference/
pipeline_comp_dir = os.path.join(parent_dir, 'pipeline_comp')  # inference/pipeline_comp/

sys.path.insert(0, parent_dir)
sys.path.insert(0, pipeline_comp_dir)


import xml.etree.ElementTree as ET
import glob
import cv2
import numpy as np
from tqdm import tqdm

from instance_funcs import *
from models.load import *
# from scipy.optimize import linear_sum_assignment
from models.lit_model import Model
from yolo_clf_new import YOLOClf

print("modules loaded successfully")

# Model paths

# Flag for handling new vs old model outputs
# True: raw argmax for newly trained models (0=none,1=single,2=double,3=triple)
# False: legacy +1/4→0 remapping for old models
RAW_MODE = True

# rm_preds = glob.glob("/ssd_scratch/Video_set_1/instance_crops_pred_new/*.txt")
rm_preds = glob.glob("/ssd_scratch/sai.teja/txt_assoc_default/*_data.txt")

assoc_weigth_path = "/archive/sai.teja/SAC_models/lr0p01_md_data_2/weights/best_compat.pt"
clf_weights = "/data3/sai.teja/triple_violations/checkpoints_ogsplit_default/weighted/tr_clf_default_puf.ckpt"

#check if the above folders exist, else exit with an error
if not os.path.exists(assoc_weigth_path) or not os.path.exists(clf_weights):
    print("Error: One or more model weight paths do not exist.")
    sys.exit(1)

if len(rm_preds) == 0:
    print("Error: No prediction files found in the specified directory.")
    sys.exit(1)

out_annots_folder = "/ssd_scratch/sai.teja/rm_preds_annots_default_puf"
os.makedirs(out_annots_folder, exist_ok=True)

out_txt_preds = "/ssd_scratch/sai.teja/triple_riding/videoset1/clf_pred_weighted_default_puf/"
os.makedirs(out_txt_preds, exist_ok=True)

def get_intersection_cost_matrix(riders, motorycles):
    """
    Returns a matrix of intersection percentages between motorycles and riders.
    The matrix is a list of lists, where the i-th row and j-th column
    contains the percentage of the motor i that intersects with the rider j.
    """
    intersection_percentage_matrix = []
    for rider in riders:
        intersection_percentages = []
        for motor in motorycles:
            r_xmin, r_ymin, r_xmax, r_ymax = rider[0], rider[1], rider[2], rider[3]
            h_xmin, h_ymin, h_xmax, h_ymax = motor[0], motor[1], motor[2], motor[3]
            intersection_area = max(0, min(r_xmax, h_xmax) - max(r_xmin, h_xmin)) * max(0, min(r_ymax, h_ymax) - max(r_ymin, h_ymin))
            h_area = (h_xmax - h_xmin) * (h_ymax - h_ymin)
            intersection_percentage = intersection_area / h_area
            intersection_percentages.append(1-intersection_percentage)
        intersection_percentage_matrix.append(intersection_percentages)
    # print(intersection_percentage_matrix)
    return intersection_percentage_matrix




# Initialize classifier using YOLOClf (same as pipeline_new.py)
dt_classifier = YOLOClf(weights_path=clf_weights, rm_assoc_path=assoc_weigth_path, raw=RAW_MODE)


# out_txt_preds = "./instance_crops_newww_clf_pred_102/"


def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA['xmax'] - boxA['xmin']) * (boxA['ymax'] - boxA['ymin'])
    boxBArea = (boxB['xmax'] - boxB['xmin']) * (boxB['ymax'] - boxB['ymin'])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# Load the XML file
for idx, preds_file in enumerate(tqdm(rm_preds,desc="Processing prediction files")):
    # print(preds_file)
    
    vid_name = preds_file.split("/")[-1].split(".")[0]
    if vid_name.endswith("_data"):
        vid_name = vid_name[:-5]  # Remove "_data" suffix
    # if '32806' not in vid_name:
    #     continue
    video_path = "/nas/deepti.rawat/Wrong-side-driving/Videos/videoset1/original_videos/" + f"{vid_name}.mp4"
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    for frame_number in range(0, 2000):  # Assuming frames start from 0 and increment by 5
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    # Initialize dictionary to store motorcycle and rider information
    # print("Length of all_frames", len(all_frames))
    if len(all_frames) == 0:
        continue
    frame_data = {}
    all_data = []
    all_inf_data = {}

    # get the predictions from the txt preds_file
    rider_preds = {}
    motor_preds = {}

    with open(preds_file, 'r') as f:
        lines = f.readlines()
        # ignore the first line
        lines = lines[1:]
        for line in lines:
            if line.strip().split(' ')[1] == '0' or line.strip().split(' ')[1] == '1':
                frame_num, label, xtl, ytl, xbr, ybr, id, assoc_id = line.strip().split(' ')
            
                frame_num, label, xtl, ytl, xbr, ybr, id, assoc_id = int(frame_num), int(label), float(xtl), float(ytl), float(xbr), float(ybr), int(id), int(assoc_id)
                if label == 0:
                    if frame_num not in rider_preds:
                        rider_preds[frame_num] = []
                    rider_preds[frame_num].append({'xmin': xtl, 'ymin': ytl, 'xmax': xbr, 'ymax': ybr, 'id': id, 'assoc_id': assoc_id})
                if label == 1:
                    if frame_num not in motor_preds:
                        motor_preds[frame_num] = []
                    motor_preds[frame_num].append({'xmin': xtl, 'ymin': ytl, 'xmax': xbr, 'ymax': ybr, 'id': id, 'assoc_id': assoc_id})
    # print("Predictions read for video ", vid_name)



    for frame_num, frame in enumerate(all_frames):
        # if frame_num != 495:
        #     continue
        # if frame_num not in rider_preds or motor_preds as a key, continue
        if frame_num not in rider_preds or frame_num not in motor_preds:
            continue


   
        # associate the riders to the motorcycles using the get_intersection_cost_matrix function and change the assoc_id of the riders to the assoc_id of the associated motorcycle
        rider_boxes = np.array([[rider['xmin'], rider['ymin'], rider['xmax'], rider['ymax'], rider['id']] for rider in rider_preds[frame_num]])
        motor_boxes = np.array([[motor['xmin'], motor['ymin'], motor['xmax'], motor['ymax'], motor['id']] for motor in motor_preds[frame_num]])
        # row_ind, col_ind = linear_sum_assignment(get_intersection_cost_matrix(motor_boxes, rider_boxes))
        # print(len(motor_boxes))
        # int_mat = np.array(get_intersection_cost_matrix(rider_boxes, motor_boxes)).T
        # row_ind = np.arange(len(motor_boxes))
        # print(int_mat)
        # col_ind = np.argmin(int_mat, axis=1)
        # print(col_ind)
        # int_vals = np.min(int_mat, axis=1)
        # col_ind[int_vals > 0.8] = -1
        # print(row_ind, col_ind)
        # for i in range(len(row_ind)):
        #     motor_preds[frame_num][row_ind[i]]['assoc_id'] = row_ind[i]

        # for i in range(len(col_ind)):
        #     print(col_ind[i])
        #     if(col_ind[i] == -1):
        #         continue
            
        #     rider_preds[frame_num][col_ind[i]]['assoc_id'] = motor_preds[frame_num][row_ind[i]]['assoc_id']
        # print(rider_boxes)
        # print(motor_boxes)  
          # loop over all motorcycles in the frame, get the riders that have the same assoc_id
        for motor_dict in motor_preds[frame_num]:
            # print("Motorcycle", motor_dict)
            riders_on_motor = [rider for rider in rider_preds[frame_num] if rider['assoc_id'] == motor_dict['assoc_id'] and rider['assoc_id'] != -1 and get_iou(rider, motor_dict) > 0]
            # print("Riders on motorcycle", riders_on_motor)
            if len(riders_on_motor) == 0:
                continue

            # get the roi_xmin which is the minimum of all the xmins of the riders on the motorcycle and the xmin of the motorcycle
            roi_xmin = min(motor_dict['xmin'], min([rider['xmin'] for rider in riders_on_motor]))
            # get the roi_ymin which is the minimum of all the ymins of the riders on the motorcycle and the ymin of the motorcycle
            roi_ymin = min(motor_dict['ymin'], min([rider['ymin'] for rider in riders_on_motor]))
            # get the roi_xmax which is the maximum of all the xmaxs of the riders on the motorcycle and the xmax of the motorcycle
            roi_xmax = max(motor_dict['xmax'], max([rider['xmax'] for rider in riders_on_motor]))
            # get the roi_ymax which is the maximum of all the ymaxs of the riders on the motorcycle and the ymax of the motorcycle
            roi_ymax = max(motor_dict['ymax'], max([rider['ymax'] for rider in riders_on_motor]))

            # get the roi_width and roi_height
            roi_width = roi_xmax - roi_xmin
            roi_height = roi_ymax - roi_ymin
            roi_frame = frame[int(roi_ymin):int(roi_ymax), int(roi_xmin):int(roi_xmax)]
            frame2 = frame.copy()
            roi_frame_temp = frame2[int(roi_ymin):int(roi_ymax), int(roi_xmin):int(roi_xmax)]

            # print(roi_xmin, roi_ymin, roi_xmax, roi_ymax)
            # print(len(riders_on_motor))
            
            tid = motor_dict['id']
            lp_num="hahaha"
            
            
            if(frame_num not in all_inf_data):
                all_inf_data[frame_num] = [[label, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid,lp_num]]
            else:
                all_inf_data[frame_num].append([label, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num])
            # print(frame_num, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num)
            # Convert to RGB and use classifier (same as pipeline_new.py)
            inst_crop = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            out_pred = dt_classifier(inst_crop)
            
            all_data.append([frame_num, out_pred, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num])
    
            # annotate rider and motor on roi_frame
            for rider in riders_on_motor:
                cv2.rectangle(roi_frame_temp, (int(rider['xmin'] - roi_xmin), int(rider['ymin'] - roi_ymin)), (int(rider['xmax'] - roi_xmin), int(rider['ymax'] - roi_ymin)), (0, 0, 255), 2)
            cv2.rectangle(roi_frame_temp, (int(motor_dict['xmin'] - roi_xmin), int(motor_dict['ymin'] - roi_ymin)), (int(motor_dict['xmax'] - roi_xmin), int(motor_dict['ymax'] - roi_ymin)), (0, 255, 0), 2)

            if out_pred >= 3:
                # print ("WRONG", roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num)
                # print(len(riders_on_motor))
                # for rider in riders_on_motor:
                #     print(rider['xmin'], rider['ymin'], rider['xmax'], rider['ymax'])
                cv2.imwrite(f"{out_annots_folder}/{vid_name}_{frame_num}_{tid}_{out_pred}.png", roi_frame_temp)
                # print("ENDDDDD")
            # cv2.imwrite(f"crops/{vid_name}_{frame_num}_{tid}_{out_pred}.png", inst_crop*255)
    
    file_name = out_txt_preds + f"/{vid_name}.txt"
    with open(file_name, "w") as f:
        for inst in all_data:
            line = " ".join(map(str, inst))
            f.write(line + "\n")

    # exit(0)
    
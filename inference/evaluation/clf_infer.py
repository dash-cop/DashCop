import xml.etree.ElementTree as ET
import glob
import cv2
import numpy as np
import os
from instance_funcs import *
from models.load import *
from scipy.optimize import linear_sum_assignment

from models.lit_model import Model

# rm_preds = glob.glob("/ssd_scratch/cvit/Video_set_1/instance_crops_pred_new/*.txt")
rm_preds = glob.glob("/ssd_scratch/cvit/Video_set_1/rm_preds/*_data.txt")

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
    print(intersection_percentage_matrix)
    return intersection_percentage_matrix


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

model = give_yolo_model(num_classes=4)
model = Model(model, None, 4)
# model_to_load = torch.load("/ssd_scratch/cvit/keshav/keshav/sdc/logs/lightning_logs/version_65/checkpoints/last.ckpt")
# model_to_load = torch.load("/ssd_scratch/cvit/keshav/last.ckpt")
model_to_load = torch.load("tr_102.ckpt")
model.load_state_dict(model_to_load['state_dict'])
model.to("cuda")
model.eval()

out_annots_folder = "./clf_predictions_rm_preds_annots"
if not os.path.exists(out_annots_folder):
    os.makedirs(out_annots_folder, exist_ok=True)

out_txt_preds = "/ssd_scratch/cvit/Video_set_1/instance_crops_newww_clf_pred_102/"
out_txt_preds = "./instance_crops_newww_clf_pred_102/"
import os
if not os.path.exists(out_txt_preds):
    os.makedirs(out_txt_preds, exist_ok=True)

os.makedirs(out_txt_preds, exist_ok=True)

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
for idx, preds_file in enumerate(rm_preds):
    print(preds_file)
    vid_name = preds_file.split("/")[-1].split(".")[0].removesuffix("_data")
    # if '32806' not in vid_name:
    #     continue
    video_path = "/ssd_scratch/cvit/dashcop/Video_set_1/videos/" + f"{vid_name}.mp4"
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    for frame_number in range(0, 2000):  # Assuming frames start from 0 and increment by 5
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    # Initialize dictionary to store motorcycle and rider information
    print("Length of all_frames", len(all_frames))
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
    print("Predictions read for video ", vid_name)



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
            print("Motorcycle", motor_dict)
            riders_on_motor = [rider for rider in rider_preds[frame_num] if rider['assoc_id'] == motor_dict['assoc_id'] and rider['assoc_id'] != -1 and get_iou(rider, motor_dict) > 0]
            print("Riders on motorcycle", riders_on_motor)
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
            print(roi_xmin, roi_ymin, roi_xmax, roi_ymax)
            print(len(riders_on_motor))
            
            tid = motor_dict['id']
            lp_num="hahaha"
            
            
            if(frame_num not in all_inf_data):
                all_inf_data[frame_num] = [[label, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid,lp_num]]
            else:
                all_inf_data[frame_num].append([label, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num])
            print(frame_num, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num)
            inst_crop = cv2.resize(roi_frame / 255, (224, 224))
            # inst_crop_ = (inst_crop - mean[None, None]) / std[None, None]
            crop = torch.from_numpy(inst_crop.astype(np.float32)).permute(2, 0, 1)[None].to("cuda")
            out_pred = torch.argmax(model(crop)[0]).item() + 1
            if(out_pred == 4):
                out_pred = 0
            all_data.append([frame_num, out_pred, roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num])
    
            # annotate rider and motor on roi_frame
            for rider in riders_on_motor:
                cv2.rectangle(roi_frame_temp, (int(rider['xmin'] - roi_xmin), int(rider['ymin'] - roi_ymin)), (int(rider['xmax'] - roi_xmin), int(rider['ymax'] - roi_ymin)), (0, 0, 255), 2)
            cv2.rectangle(roi_frame_temp, (int(motor_dict['xmin'] - roi_xmin), int(motor_dict['ymin'] - roi_ymin)), (int(motor_dict['xmax'] - roi_xmin), int(motor_dict['ymax'] - roi_ymin)), (0, 255, 0), 2)

            if out_pred >= 3:
                print ("WRONG", roi_xmin,roi_ymin,roi_xmax,roi_ymax, tid, lp_num)
                print(len(riders_on_motor))
                for rider in riders_on_motor:
                    print(rider['xmin'], rider['ymin'], rider['xmax'], rider['ymax'])
                cv2.imwrite(f"{out_annots_folder}/{vid_name}_{frame_num}_{tid}_{out_pred}.png", roi_frame_temp)
                print("ENDDDDD")
            # cv2.imwrite(f"crops/{vid_name}_{frame_num}_{tid}_{out_pred}.png", inst_crop*255)
    
    file_name = out_txt_preds + f"/{vid_name}.txt"
    with open(file_name, "w") as f:
        for inst in all_data:
            line = " ".join(map(str, inst))
            f.write(line + "\n")

    # exit(0)
    
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import cv2
from instance_funcs import *

CUDA_VISIBLE_DEVICES=0
hnh_model = YOLO('./hnh_frame_train/weights/best.pt')
videos_folder = '../../Video_set_1/videos/'
rm_preds_txt_files_folder = '../../Video_set_1/rm_preds/'

def get_intersection_cost_matrix(riders, helmets):
    """
    Returns a matrix of intersection percentages between helmets and riders.
    The matrix is a list of lists, where the i-th row and j-th column
    contains the percentage of the helmet i that intersects with the rider j.
    """
    intersection_percentage_matrix = []
    for rider in riders:
        intersection_percentages = []
        for helmet in helmets:
            r_xmin, r_ymin, r_xmax, r_ymax = rider[0], rider[1], rider[2], rider[3]
            h_xmin, h_ymin, h_xmax, h_ymax = helmet[0], helmet[1], helmet[2], helmet[3]
            intersection_area = max(0, min(r_xmax, h_xmax) - max(r_xmin, h_xmin)) * max(0, min(r_ymax, h_ymax) - max(r_ymin, h_ymin))
            h_area = (h_xmax - h_xmin) * (h_ymax - h_ymin)
            intersection_percentage = intersection_area / h_area
            intersection_percentages.append(1-intersection_percentage)
        intersection_percentage_matrix.append(intersection_percentages)
    print(intersection_percentage_matrix)
    return intersection_percentage_matrix


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



# hnh_model.to("cuda")
# loop over videos in triple_reviewed/videos

# create hnh_preds_frame dir if it does not exist, else delete all files in it
if not os.path.exists('./hnh_preds_frame'):
    os.makedirs('./hnh_preds_frame')
else:
    for file in os.listdir('./hnh_preds_frame'):
        os.remove(os.path.join('./hnh_preds_frame', file))

for video_name in os.listdir(videos_folder):

    pred_txt_file_path = rm_preds_txt_files_folder + video_name.split('.')[0] + '_data.txt'
    if not os.path.exists(pred_txt_file_path):
        print("rm preds do not exist for video ", video_name)
        continue

    all_frames = []
    all_frames_copy = []
    # read frames from the video
    cap = cv2.VideoCapture(videos_folder + video_name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    # read frames from the video
    cap = cv2.VideoCapture(videos_folder + video_name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames_copy.append(frame)

    # get the predictions from the txt file
    rider_preds = {}
    motor_preds = {}

    with open(pred_txt_file_path, 'r') as f:
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
    print("Predictions read for video ", video_name)
    # print("Rider preds: ", rider_preds)
    # print("Motor preds: ", motor_preds)
    
    f.close()

    # create hnh_annots_frame/video_name dir if it does not exist, else delete all files in it
    if not os.path.exists('./hnh_annots_frame/' + video_name.split('.')[0]):
        os.makedirs('./hnh_annots_frame/' + video_name.split('.')[0])
    else:
        for file in os.listdir('./hnh_annots_frame/' + video_name.split('.')[0]):
            os.remove(os.path.join('./hnh_annots_frame/' + video_name.split('.')[0], file))

    # create a txt file to write the detections to as hnh_preds_frame/video_name.txt
    hnh_out_file = open('./hnh_preds_frame/' + video_name.split('.')[0] + '.txt', 'w')

    # add frame width and height to the file
    hnh_out_file.write(str(all_frames[0].shape[1]) + ',' + str(all_frames[0].shape[0]) + '\n')

    # all_frames = all_frames[:11]

    for frame_num, frame in enumerate(all_frames):
        # convert the frame to RGB
        frame_copy = cv2.cvtColor(all_frames_copy[frame_num], cv2.COLOR_BGR2RGB)


        # if frame_num not in rider_preds or motor_preds as a key, continue
        if frame_num not in rider_preds or frame_num not in motor_preds:
            continue

        # associate the riders to the motorcycles using the get_intersection_cost_matrix function and change the assoc_id of the riders to the assoc_id of the associated motorcycle
        rider_boxes = np.array([[rider['xmin'], rider['ymin'], rider['xmax'], rider['ymax'], rider['id']] for rider in rider_preds[frame_num]])
        motor_boxes = np.array([[motor['xmin'], motor['ymin'], motor['xmax'], motor['ymax'], motor['id']] for motor in motor_preds[frame_num]])
        row_ind, col_ind = linear_sum_assignment(get_intersection_cost_matrix(motor_boxes, rider_boxes))
        for i in range(len(col_ind)):
            rider_preds[frame_num][col_ind[i]]['assoc_id'] = motor_preds[frame_num][row_ind[i]]['assoc_id']

        preds = hnh_model(frame, classes = [0,1], conf=0.4)[0]
        # Get the detections
        hnh_boxes, hnh_scores, hnh_classes = getDetections(preds, frame)
        
        helmet_boxes = hnh_boxes[hnh_classes == 0]

        no_helmet_boxes = hnh_boxes[hnh_classes == 1]

        # Bounding boxes are in normalized ymin, xmin, ymax, xmax
        original_h, original_w, _ = frame.shape

        # The tracker will accept boxes in the format (xc, yc, w, h)
        helmet_boxes = format_boxes_new_xyxy(helmet_boxes, original_h, original_w)
        no_helmet_boxes = format_boxes_new_xyxy(no_helmet_boxes, original_h, original_w)

        all_hnh_boxes = [] # list of all helmet and no helmet boxes in form xmin, ymin, xmax, ymax,label
        for box in helmet_boxes:
            all_hnh_boxes.append([box[0], box[1], box[2], box[3], 0])
        for box in no_helmet_boxes:
            all_hnh_boxes.append([box[0], box[1], box[2], box[3], 1])
        
        # assign helmets in all_hnh to riders in riders_on_motor using linear sum assignment (based on intersection of helmet and rider boxes)
        if len(all_hnh_boxes) > 0:
            helmet_boxes = np.array(all_hnh_boxes)
            rider_boxes = np.array([[rider['xmin'], rider['ymin'], rider['xmax'], rider['ymax'], rider['id']] for rider in rider_preds[frame_num]])
            row_ind, col_ind = linear_sum_assignment(get_intersection_cost_matrix(rider_boxes, helmet_boxes))
            print(row_ind, col_ind)
            for rider_idx, rider in enumerate(rider_boxes): 
                # if rider_idx is in row_idx then assign the helmet to the rider
                r_xmin = rider[0]
                r_ymin = rider[1]
                r_xmax = rider[2]
                r_ymax = rider[3]

                rider_helmet_assigned = False

                if rider_idx in row_ind:
                    idx = 0
                    for i in range(len(row_ind)):
                        if row_ind[i] == rider_idx:
                            idx = i
                            break
                    helmet_box = helmet_boxes[col_ind[idx]]
                    intersection_area = max(0, min(rider[2], helmet_box[2]) - max(rider[0], helmet_box[0])) * max(0, min(rider[3], helmet_box[3]) - max(rider[1], helmet_box[1]))
                    helmet_area = (helmet_box[2] - helmet_box[0]) * (helmet_box[3] - helmet_box[1])
                    intersection_percentage = intersection_area / helmet_area
                    if intersection_percentage > 0.5:
                        rider_helmet_assigned = True

                        # write the detections to the file : frame_num, rider_id, r_xmin, r_ymin, r_xmax, r_ymax, hnh_label, helmet_xmin, helmet_ymin, helmet_xmax, helmet_ymax
                        hnh_out_file.write(str(frame_num) + ',' + str(rider[4]) + ',' + str(r_xmin) + ',' + str(r_ymin) + ',' + str(r_xmax) + ',' + str(r_ymax) + ',' + str(helmet_box[4]) + ',' + str(helmet_box[0]) + ',' + str(helmet_box[1]) + ',' + str(helmet_box[2]) + ',' + str(helmet_box[3]) + '\n')
                        color = (0, 255, 0) if helmet_box[4] == 0 else (255, 0, 0)
                        frame_copy = cv2.rectangle(frame_copy, (int(helmet_box[0]), int(helmet_box[1])), (int(helmet_box[2]), int(helmet_box[3])), color, 2)

                if not rider_helmet_assigned:
                    # write the detections to the file : frame_num, rider_id, r_xmin, r_ymin, r_xmax, r_ymax, -1, -1, -1, -1, -1
                    hnh_out_file.write(str(frame_num) + ',' + str(rider[4]) + ',' + str(r_xmin) + ',' + str(r_ymin) + ',' + str(r_xmax) + ',' + str(r_ymax) + ',-1,-1,-1,-1,-1\n')

                frame_copy = cv2.rectangle(frame_copy, (int(r_xmin), int(r_ymin)), (int(r_xmax), int(r_ymax)), (0, 0, 255), 2)

        else:
            for rider in rider_preds[frame_num]:
                r_xmin = rider['xmin']
                r_ymin = rider['ymin']
                r_xmax = rider['xmax']
                r_ymax = rider['ymax']
                # write the detections to the file : frame_num, rider_id, r_xmin, r_ymin, r_xmax, r_ymax, -1, -1, -1, -1, -1
                hnh_out_file.write(str(frame_num) + ',' + str(rider['id']) + ',' + str(r_xmin) + ',' + str(r_ymin) + ',' + str(r_xmax) + ',' + str(r_ymax) + ',-1,-1,-1,-1,-1\n')
                frame_copy = cv2.rectangle(frame_copy, (int(r_xmin), int(r_ymin)), (int(r_xmax), int(r_ymax)), (0, 0, 255), 2)

        # save the frame with the detections
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./hnh_annots_frame/' + video_name.split('.')[0] + '/' + str(frame_num) + '.jpg', frame_copy)
        print(f'Frame {frame_num} of video {video_name} done')

    hnh_out_file.close()
    # break

    # with open(pred_txt_file_path, 'r') as f:
    #     # write the detections to the file : frame, label, xtl, ytl, xbr, ybr, score
    #     for i in range(len(helmet_boxes)):
    #         out_file.write(str(frame_num) + ',rider,' + str(float(rider_boxes[i][0])) + ',' + str(float(rider_boxes[i][1])) + ',' + str(float(rider_boxes[i][2])) + ',' + str(float(rider_boxes[i][3])) + ',' + str(float(rider_scores[i])) + '\n')

    #     for i in range(len(no_helmet_boxes)):
    #         out_file.write(str(frame_num) + ',motorcycle,' + str(float(motor_boxes[i][0])) + ',' + str(float(motor_boxes[i][1])) + ',' + str(float(motor_boxes[i][2])) + ',' + str(float(motor_boxes[i][3])) + ',' + str(float(motor_scores[i])) + '\n')

    #     print(f'Frame {frame_num} of video {mots_folder} done')

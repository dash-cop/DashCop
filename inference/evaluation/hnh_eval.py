import xml.etree.ElementTree as ET
import os
import sys
import cv2

gt_folder = 'Video_set_1/gt_annots'
preds_folder = './hnh_instance_inference/hnh_preds_frame'
rider_area_threshold =  0.008
helmet_area_threshold = 0.00005

mots_folders = []
videos_names = []

# loop over all folders in gt_annots
for folder in os.listdir(gt_folder):
    # if folder name is not zip file
    if not folder.endswith('.zip'):
        mots_folders.append(os.path.join(gt_folder, folder))
        videos_names.append(folder)

print('MOTS folders:', mots_folders)
print('Videos names:', videos_names)


for folder in mots_folders:
    for filename in os.listdir(folder):
        if filename.endswith("annotations.xml"):
            # create a txt file rm_annotations.txt to store the annotations
            if(os.path.exists(folder + '/rm_annotations.txt') and os.path.exists(folder + '/annotations_helmet.txt')):
                continue
            out_file = open(folder + '/rm_annotations.txt', 'w')
            out_file_helmet = open(folder + '/annotations_helmet.txt', 'w')
            tree = ET.parse(os.path.join(folder, filename))
            root = tree.getroot()
            # loop over all track tags
            for track in root.iter('track'):
                if track.attrib['label'] == 'rider':
                    # loop over all box tags
                    for box in track.iter('box'):
                        # get the attributes of the box tag
                        if box.attrib['outside'] == '0' and box.attrib['occluded'] == '0':
                            frame = box.attrib['frame']
                            xtl = box.attrib['xtl']
                            ytl = box.attrib['ytl']
                            xbr = box.attrib['xbr']
                            ybr = box.attrib['ybr']
                            # write the attributes to the txt file
                            out_file.write(frame + ',' + track.attrib['id'] + ',' + track.attrib['label'] + ',' + xtl + ',' + ytl + ',' + xbr + ',' + ybr + '\n')

                # for helmets
                if track.attrib['label'] == 'helmet' or track.attrib['label'] == 'no_helmet':
                    # loop over all box tags
                    for box in track.iter('box'):
                        # get the attributes of the box tag
                        if box.attrib['outside'] == '0' and box.attrib['occluded'] == '0':
                            frame = box.attrib['frame']
                            xtl = box.attrib['xtl']
                            ytl = box.attrib['ytl']
                            xbr = box.attrib['xbr']
                            ybr = box.attrib['ybr']
                            # write the attributes to the txt file
                            out_file_helmet.write(frame + ',' +  track.attrib['label'] + ',' + xtl + ',' + ytl + ',' + xbr + ',' + ybr + '\n')



def annotate(frame, label, xtl, ytl, xbr, ybr,color=None, txt = None):
    if color is not None:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
    elif label == 'motorcycle':
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 215, 0), 2)
    elif label == 'rider':
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (100, 200, 235), 2)
    elif label == 'rm-instance':
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    elif label == 'helmet':
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
    elif label == 'no_helmet':
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)
    # put the line number on the frame
    if txt is not None:
        cv2.putText(frame, str(txt), (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

def get_matches(boxes1, boxes2, thresh = 0.7, use_iou = False):
    matches = []
    fp = []
    fn = []
    
    # calculate the intersection area between each pair of boxes of boxes1 and boxes2. It will be an m x n matrix where m is the number of boxes in boxes1 and n is the number of boxes in boxes2.
    intersection_matrix = []
    for box1 in boxes1:
        intersection_row = []
        for box2 in boxes2:
            # calculate the intersection area
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            if use_iou:
                iou = intersection_area / ((box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) - intersection_area)
                intersection_row.append(iou)
            else:
                # divide the intersection area by the area of the box1
                intersection_area = intersection_area / ((box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1))
                intersection_row.append(intersection_area)
        intersection_matrix.append(intersection_row)

    # print(intersection_matrix)

    # perform matching in the intersection_matrix such that the sum of the intersection areas of the matched pairs is maximized
    while True:
        max_intersection = 0
        max_intersection_index = (0, 0)
        for i in range(len(intersection_matrix)):
            for j in range(len(intersection_matrix[i])):
                if intersection_matrix[i][j] > max_intersection:
                    max_intersection = intersection_matrix[i][j]
                    max_intersection_index = (i, j)
        if max_intersection < thresh:
            break
        matches.append((max_intersection_index[0], max_intersection_index[1]))
        # set the intersection areas of the matched boxes to 0
        for i in range(len(intersection_matrix)):
            intersection_matrix[i][max_intersection_index[1]] = 0
        for j in range(len(intersection_matrix[max_intersection_index[0]])):
            intersection_matrix[max_intersection_index[0]][j] = 0

    # get the false positives and false negatives
    for i in range(len(boxes1)):
        if i not in [match[0] for match in matches]:
            fp.append(boxes1[i])
    for j in range(len(boxes2)):
        if j not in [match[1] for match in matches]:
            fn.append(boxes2[j])

    return matches, fp, fn


all_helmets_frame_gt = [] # array of M,5 : class, x1, y1, x2, y2
total_helmets_gt = 0
total_riders_gt = 0
total_matches_gt = 0
discarded_helmets_gt = 0
discarded_riders_gt = 0
all_riders_frame_gt = [] # array of M,5 : class, x1, y1, x2, y2

# for each frame, get the list of helmet boxes and the list of rider boxes from the annotations.txt file and annotations_helmet.txt file
for folder in mots_folders:
    video_name = folder.split('/')[-1]
    try : 
        rider_helmet_file = open(os.path.join(preds_folder, video_name + '.txt'), 'r')
        # read the first line of the file as it contains the frame size
        rider_helmet = rider_helmet_file.readlines()
        rider_helmet_file.close()
        # read first line as frame width and height
        frame_width, frame_height = rider_helmet[0].split(',')
        frame_width, frame_height = int(float(frame_width)), int(float(frame_height))
        print("frame_width: ", frame_width)
        print("frame_height: ", frame_height)
        frame_area = frame_width * frame_height
    except:
        print("file not found", video_name)
        continue


    out_file_rider_helmet = open(folder + '/rider_helmet.txt', 'w')
    helmets_without_riders = 0
    riders_without_helmets = 0
    far_riders_discarded = 0
    num_matches = 0
    # read the annotations.txt file
    annotations_file = open(folder + '/rm_annotations.txt', 'r')
    annotations = annotations_file.readlines()
    annotations_file.close()
    # read the annotations_helmet.txt file
    annotations_helmet_file = open(folder + '/annotations_helmet.txt', 'r')
    annotations_helmet = annotations_helmet_file.readlines()
    annotations_helmet_file.close()
    
    for frame_num in range(0, 2000):

        helmets_frame_gt = [] # array of M,5 : class, x1, y1, x2, y2
        helmets_frame_preds = [] # array of N,5 : x1,y1,x2,y2,conf,class where conf = 1
        riders_frame_gt = [] # array of M,5 : class, x1, y1, x2, y2

        # get the list of helmet boxes for the frame
        helmet_boxes = []
        for annotation_helmet in annotations_helmet:
            frame, label, xtl, ytl, xbr, ybr = annotation_helmet.split(',')
            if label == 'helmet':
                label = 0
            elif label == 'no_helmet':
                label = 1
            if int(frame) == frame_num:
                helmet_boxes.append([float(xtl), float(ytl), float(xbr), float(ybr), label])
                area_helmet = (float(xbr) - float(xtl)) * (float(ybr) - float(ytl))
                if area_helmet/frame_area > helmet_area_threshold:
                    helmets_frame_gt.append([label, float(xtl), float(ytl), float(xbr), float(ybr)])
                    total_helmets_gt += 1
                else:
                    discarded_helmets_gt += 1
        # get the list of rider boxes for the frame
        rider_boxes = []
        for annotation in annotations:
            frame, track_id, label, xtl, ytl, xbr, ybr = annotation.split(',')
            if int(frame) == frame_num:
                rider_boxes.append([float(xtl), float(ytl), float(xbr), float(ybr), track_id])
                riders_frame_gt.append([0, float(xtl), float(ytl), float(xbr), float(ybr)])
                total_riders_gt += 1
            
        # for each helmet box, check if it is inside a rider box
        matches,fp,fn = get_matches(helmet_boxes, rider_boxes)

        # frame = cv2.imread(folder + '/images/frame_' + str(frame_num).zfill(6) + '.png')
        for match in matches:

            rider_bbox_area = (rider_boxes[match[1]][2] - rider_boxes[match[1]][0]) * (rider_boxes[match[1]][3] - rider_boxes[match[1]][1])
            # print("rider_bbox_area: ", rider_bbox_area)
            # print("frame_area: ", frame_area)
            if rider_bbox_area/frame_area > rider_area_threshold:
                out_file_rider_helmet.write(str(frame_num) + ',' + str(rider_boxes[match[1]][4]) + ',' + str(rider_boxes[match[1]][0]) + ',' + str(rider_boxes[match[1]][1]) + ',' + str(rider_boxes[match[1]][2]) + ',' + str(rider_boxes[match[1]][3]) + ',' + str(helmet_boxes[match[0]][4]) + '\n')
                # frame = annotate(frame, 'rider', int(rider_boxes[match[1]][0]), int(rider_boxes[match[1]][1]), int(rider_boxes[match[1]][2]), int(rider_boxes[match[1]][3]))
                hnh_label = ''
                if helmet_boxes[match[0]][4] == 0:
                    hnh_label = 'helmet'
                elif helmet_boxes[match[0]][4] == 1:
                    hnh_label = 'no_helmet'
                # frame = annotate(frame, hnh_label, int(helmet_boxes[match[0]][0]), int(helmet_boxes[match[0]][1]), int(helmet_boxes[match[0]][2]), int(helmet_boxes[match[0]][3]))
            else:
                far_riders_discarded += 1

        helmets_without_riders += len(fp)
        num_matches += len(matches)
        all_helmets_frame_gt.append(helmets_frame_gt)
        all_riders_frame_gt.append(riders_frame_gt)
    out_file_rider_helmet.close()
    print("folder: ", folder)
    print("total matches: ", num_matches - far_riders_discarded)
    total_matches_gt += num_matches - far_riders_discarded
    print("helmets_without_riders: ", helmets_without_riders)
    print("riders_without_helmets: ", riders_without_helmets)
    print("far_riders_discarded: ", far_riders_discarded)
    discarded_riders_gt += far_riders_discarded

print("total_helmets_gt: ", total_helmets_gt)
print("discarded_helmets_gt: ", discarded_helmets_gt)
print("total_riders_gt: ", total_riders_gt)
print("total_matches_gt: ", total_matches_gt)
print("discarded_riders_gt: ", discarded_riders_gt)

# get the predicted annotations in rider_helmet_predictions.txt
# ground truth is rider_helmet.txt
import numpy as np
total_fp = 0
fp_det_tracks = {}
total_fn = 0
fn_det_tracks = {}
total_matches = 0
gt_helmet_tracks = {}
gt_no_helmet_tracks = {}
all_folders_gt_to_pred = {}
all_gt_tracks = {}
all_pred_tracks = {}
total_pred_riders = 0
far_riders_discarded_pred = 0
total_gt_helmets_tracks = 0
total_gt_no_helmets_tracks = 0
total_pred_helmets_tracks = 0
total_pred_no_helmets_tracks = 0
for idx, folder in enumerate(mots_folders):
    # read the predicted annotations
    try:
        with open(preds_folder + '/' + videos_names[idx] + '.txt', 'r') as f:
            lines_pred = f.readlines()
    except:
        print("file not found", videos_names[idx])
        continue
    # remove the first line
    lines_pred.pop(0)
    # remove the last line if it is empty
    if lines_pred[-1] == '':
        lines_pred.pop(-1)
    # read the ground truth annotations
    with open(folder + '/rider_helmet.txt', 'r') as f:
        lines_gt = f.readlines()
    tracks_gt_to_pred = {}
    fp_det_tracks[folder] = []
    fn_det_tracks[folder] = []
    all_gt_tracks[folder] = []
    all_pred_tracks[folder] = []
    gt_helmet_track_ids = []
    gt_no_helmet_track_ids = []
    pred_helmet_track_ids = []
    pred_no_helmet_track_ids = []
    for frame_num in range(0, 2000):

        gt_tracks = []
        pred_tracks = []
        for line in lines_gt:
            line = line.split(',')
            if int(line[0]) == frame_num:
                xmin, ymin, xmax, ymax, id, label = float(line[2]), float(line[3]), float(line[4]), float(line[5]), int(line[1]), int(line[6])
                gt_tracks.append([xmin, ymin, xmax, ymax, id, label])
                if label == 0:
                    gt_helmet_track_ids.append(id)
                elif label == 1:
                    gt_no_helmet_track_ids.append(id)

        for line in lines_pred:
            line = line.split(',')
            if int(line[0]) == frame_num:
                xmin, ymin, xmax, ymax, id, label = float(line[2]), float(line[3]), float(line[4]), float(line[5]), int(float(line[1])), int(float(line[6]))
                rider_area = (xmax - xmin) * (ymax - ymin)
                if rider_area/frame_area > rider_area_threshold:
                    pred_tracks.append([xmin, ymin, xmax, ymax, id, label])
                    total_pred_riders += 1
                    if label == 0:
                        pred_helmet_track_ids.append(id)
                    elif label == 1:
                        pred_no_helmet_track_ids.append(id)
                else:
                    far_riders_discarded_pred += 1

                
        # helmets need to be read from helmets.txt and added here as well

        matches, fp, fn = get_matches(gt_tracks, pred_tracks, thresh = 0.5, use_iou = True)
        total_fp += len(fp)
        total_fn += len(fn)
        # add the false positives and false negatives ids to the list of detection tracks
        for track in fp:
            # add if it is not already in the list
            if track[4] not in fp_det_tracks[folder]:
                fp_det_tracks[folder].append(track[4])
        for track in fn:
            # add if it is not already in the list
            if track[4] not in fn_det_tracks[folder]:
                fn_det_tracks[folder].append(track[4])
        for match in matches:
            # if no key exists for the gt track id, create a key with the gt track id and the value as a list containing the pred track id
            if gt_tracks[match[0]][4] not in tracks_gt_to_pred:
                tracks_gt_to_pred[gt_tracks[match[0]][4]] = [pred_tracks[match[1]][4]]
            else:
                tracks_gt_to_pred[gt_tracks[match[0]][4]].append(pred_tracks[match[1]][4])
        total_matches += len(matches)

        # concatenate the gt_tracks and pred_tracks to all_gt_tracks and all_pred_tracks
        all_gt_tracks[folder] += gt_tracks
        all_pred_tracks[folder] += pred_tracks

    total_gt_helmets_tracks += len(np.unique(gt_helmet_track_ids))
    total_gt_no_helmets_tracks += len(np.unique(gt_no_helmet_track_ids))
    total_pred_helmets_tracks += len(np.unique(pred_helmet_track_ids))
    total_pred_no_helmets_tracks += len(np.unique(pred_no_helmet_track_ids))

    print("folder: ", folder)
    print("tracks_gt_to_pred: ", tracks_gt_to_pred)
    all_folders_gt_to_pred[folder] = tracks_gt_to_pred
    # remove the tracks from fp_det_tracks and fn_det_tracks that are in the matches
    for match in matches:
        if pred_tracks[match[1]][4] in fp_det_tracks[folder]:
            fp_det_tracks[folder].remove(pred_tracks[match[1]][4])
        if gt_tracks[match[0]][4] in fn_det_tracks[folder]:
            fn_det_tracks[folder].remove(gt_tracks[match[0]][4])

print("fp_det_tracks: ", fp_det_tracks)
print("fn_det_tracks: ", fn_det_tracks)

total_fp_instances = 0
total_fn_instances = 0
for folder in fp_det_tracks:
    total_fp_instances += len(fp_det_tracks[folder])
    total_fn_instances += len(fn_det_tracks[folder])
print("total_fp_instances: ", total_fp_instances)
print("total_fn_instances: ", total_fn_instances)
print("total_matche instances: ", total_matches)
print("total gt helmet rider tracks: ", total_gt_helmets_tracks)
print("total gt no helmet rider tracks: ", total_gt_no_helmets_tracks)
print("total pred riders: ", total_pred_riders)
print("far_riders_discarded_pred: ", far_riders_discarded_pred)
print("total pred helmet rider tracks: ", total_pred_helmets_tracks)
print("total pred no helmet rider tracks: ", total_pred_no_helmets_tracks)

# if it does not exist, create a folder challan_rate_calc_input, else remove all files in the folder
if not os.path.exists('challan_rate_calc_input'):
    os.makedirs('challan_rate_calc_input')
else:
    for subdir in os.listdir('challan_rate_calc_input'):
        for filename in os.listdir('challan_rate_calc_input/' + subdir):
            os.remove('challan_rate_calc_input/' + subdir + '/' + filename)
            
# for every gt box in tracks_gt_to_pred, check the label of the corresponding pred box id that occurs the most number of times in tracks_gt_to_pred[gt_box_id]
id_switich_count = 0
correct_instances_helmet = 0
total_instances_helmet = 0
correct_instances_no_helmet = 0
total_instances_no_helmet = 0
for folder in mots_folders:
    if folder not in all_folders_gt_to_pred:
        print("folder not in preds: ", folder)
        continue
    tracks_gt_to_pred = all_folders_gt_to_pred[folder]


    # create a folder in challan_rate_calc_input with the name of the folder
    if not os.path.exists('challan_rate_calc_input/' + folder.split('/')[-1]):
        os.makedirs('challan_rate_calc_input/' + folder.split('/')[-1])
    else:
        for filename in os.listdir('challan_rate_calc_input/' + folder.split('/')[-1]):
            os.remove('challan_rate_calc_input/' + folder.split('/')[-1] + '/' + filename)

    unique_matched_pred_tracks = []
    # make a file matched_ids in the folder and loop over gt_to_preds and write the gt id
    with open('challan_rate_calc_input/' + folder.split('/')[-1] + '/matched_ids.txt', 'w') as f:
        
        gt_to_preds = all_folders_gt_to_pred[folder]
        
        for gt_box_id in tracks_gt_to_pred:
            pred_box_ids = tracks_gt_to_pred[gt_box_id]
            pred_box_id_counts = {}
            for pred_box_id in pred_box_ids:
                if pred_box_id not in pred_box_id_counts:
                    pred_box_id_counts[pred_box_id] = 1
                else:
                    pred_box_id_counts[pred_box_id] += 1
            max_count = 0
            max_count_pred_box_id = 0
            for pred_box_id in pred_box_id_counts:
                if pred_box_id_counts[pred_box_id] > max_count:
                    max_count = pred_box_id_counts[pred_box_id]
                    max_count_pred_box_id = pred_box_id


            # get the label of the pred box id that occurs the most number of times
            pred_label_counts = {}
            for pred_track in all_pred_tracks[folder]:
                # print("pred_track: ", pred_track)
                if pred_track[4] == max_count_pred_box_id:
                    if pred_track[5] not in pred_label_counts:
                        pred_label_counts[pred_track[5]] = 1
                    else:
                        pred_label_counts[pred_track[5]] += 1

            max_count_pred_label = 0
            max_count = 0
            for pred_label in pred_label_counts:
                if pred_label_counts[pred_label] > max_count:
                    max_count = pred_label_counts[pred_label]
                    max_count_pred_label = pred_label
            # print(pred_label_counts)
            id_switich_count += len(pred_box_id_counts.keys()) - 1
            if(id_switich_count > 0):
                print("pred_label_counts: ", pred_box_id_counts)
            # print("gt_box_id: ", gt_box_id, "pred_label: ", max_count_pred_label)
            # find a track with gt_box_id in gt_tracks and get label
            gt_label = 0
            for gt_track in all_gt_tracks[folder]:
                if gt_track[4] == gt_box_id:
                    gt_label = gt_track[5]
            if gt_label == max_count_pred_label:
                if gt_label == 0:
                    correct_instances_helmet += 1
                else:
                    correct_instances_no_helmet += 1
                
            if gt_label == 0:
                total_instances_helmet += 1
            else:
                total_instances_no_helmet += 1

            string_pred_box_ids = ''
            for pred_box_id in np.unique(pred_box_ids):
                string_pred_box_ids += str(pred_box_id) + ','
            string_pred_box_ids = string_pred_box_ids[:-1]
            # write the gt id, gt_label, max_count_pred_label, unique list of pred box ids
            f.write(str(gt_box_id) + ',' + str(gt_label) + ',' + str(max_count_pred_label) + ',' + string_pred_box_ids + '\n')

            unique_matched_pred_tracks += np.unique(pred_box_ids).tolist()

    f.close()

    gt_tracks = all_gt_tracks[folder]
    false_negative_tracks = []
    for gt_track in gt_tracks:
        if gt_track[4] not in tracks_gt_to_pred:
            false_negative_tracks.append(gt_track)
    mantain_unique_fn_tracks = []
    # create the file false_negative_background_tracks.txt
    with open('challan_rate_calc_input/' + folder.split('/')[-1] + '/false_negative_background_tracks.txt', 'w') as f:
        for track in false_negative_tracks:
            if track[4] in mantain_unique_fn_tracks:
                continue
            f.write(str(track[4]) + ',' + str(track[5]) + '\n')
            mantain_unique_fn_tracks.append(track[4])
    f.close()

    # create the file false_positive_tracks.txt
    pred_tracks = all_pred_tracks[folder]
    false_positive_tracks = []
    for pred_track in pred_tracks:
        if pred_track[4] not in unique_matched_pred_tracks:
            false_positive_tracks.append(pred_track)
    mantain_unique_fp_tracks = []
    with open('challan_rate_calc_input/' + folder.split('/')[-1] + '/false_positive_tracks.txt', 'w') as f:
        for track in false_positive_tracks:
            if int(track[5]) == -1:
                continue
            if track[4] in mantain_unique_fp_tracks:
                continue
            f.write(str(track[4]) + ',' + str(track[5]) + '\n')
            mantain_unique_fp_tracks.append(track[4])
    f.close()

print("id_switich_count: ", id_switich_count)
print("correct_instances_helmet: ", correct_instances_helmet)
print("total_instances_helmet: ", total_instances_helmet)
print("correct_instances_no_helmet: ", correct_instances_no_helmet)
print("total_instances_no_helmet: ", total_instances_no_helmet)

track_level_conf_mat = np.zeros((3, 3))

# True positive Helmet
track_level_conf_mat[0, 0] = correct_instances_helmet

# Miscalssified No Helmet as Helmet
track_level_conf_mat[0, 1] = total_instances_no_helmet - correct_instances_no_helmet

# Miscalssified Helmet as No Helmet
track_level_conf_mat[1, 0] = total_instances_helmet - correct_instances_helmet

# True positive No Helmet
track_level_conf_mat[1, 1] = correct_instances_no_helmet

# Helmet false positives
track_level_conf_mat[0, 2] = total_pred_helmets_tracks - track_level_conf_mat[0, 1] - track_level_conf_mat[0, 0]

# No Helmet false positives
track_level_conf_mat[1, 2] = total_pred_no_helmets_tracks - track_level_conf_mat[1, 0] - track_level_conf_mat[1, 1]

# Helmet False Negatives
track_level_conf_mat[2, 0] = total_gt_helmets_tracks - track_level_conf_mat[0, 0] - track_level_conf_mat[1, 0]

# No Helmet False Negatives
track_level_conf_mat[2, 1] = total_gt_no_helmets_tracks - track_level_conf_mat[0, 1] - track_level_conf_mat[1, 1]

print("track_level_conf_mat: ")
print(track_level_conf_mat)

import matplotlib.pyplot as plt
# confusion_mat = confusion_matrix.matrix
confusion_mat = track_level_conf_mat
# # print plot of confusion matrix using pyplot and define x labels as gt Helmet, gt Person, gt Background and y labels as pred Helmet, pred Person, pred Background
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("HNH Holistic Results", size = 30)
# show numerical values in each cell
for i in range(3):
    for j in range(3):
        plt.text(j, i, int(confusion_mat[i][j]), horizontalalignment="center", color = "white" if confusion_mat[i][j] > 1500 else "black", size = 40)
plt.xlabel("GT", size = 20)
plt.ylabel("Pred", size = 20)
plt.xticks([0,1,2], ["Helmet", "NO-Helmet", "Background"], rotation=0, size = 14)
plt.yticks([0,1,2], ["Helmet", "NO-Helmet", "Background"], rotation=0, size = 20)
# increase size of plot
plt.rcParams["figure.figsize"] = (10,7)
plt.colorbar()
plt.savefig("hnh_holistic_results.png")

tp = confusion_mat[1][1]
fp = confusion_mat[1][0] + confusion_mat[1][2]
fn = confusion_mat[0][1] + confusion_mat[2][1]

# print("Total GT: ", gt_count)
# print("Total Predicted: ", pred_det_count)
print("FOR NO-HELMET CLASS : ")
print("Precision: ", round(tp/(tp+fp), 4))
print("Recall: ", round(tp/(tp+fn),4))
print("f1-score: ", round(2*tp/(2*tp+fp+fn),4))
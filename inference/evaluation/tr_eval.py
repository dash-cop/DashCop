import os
import glob
import cv2
import numpy as np
import torch
from instance_funcs import *

all_gt_files = glob.glob("/ssd_scratch/cvit/keshav/dashcop/Video_set_1/instance_crops_gt/*.txt")
all_pred_files = ["/ssd_scratch/cvit/keshav/dashcop/Video_set_1/instance_crops_newww_clf_pred_102/" + file.split('/')[-1] for file in all_gt_files]

# if it does not exist, create a folder challan_rate_calc_input_tr, else remove all files in the folder
if not os.path.exists('challan_rate_calc_input_tr'):
    os.makedirs('challan_rate_calc_input_tr')
else:
    for subdir in os.listdir('challan_rate_calc_input_tr'):
        for filename in os.listdir('challan_rate_calc_input_tr/' + subdir):
            os.remove('challan_rate_calc_input_tr/' + subdir + '/' + filename)

max_f1_score = 0
max_thresh = 0
for inst_area_threshold in [0.014]:
    rider_area_threshold =  0.0005
    NO_TR_violation_area_threshold = 0.00005
    rider_area_threshold =  0.001
    NO_TR_violation_area_threshold = 0.0001
    # inst_area_threshold = 0.001
    frame_area = 2560 * 1440
    conf_mat = np.zeros((3, 3))

    total_fp = 0
    fp_det_tracks = {}
    total_fn = 0
    fn_det_tracks = {}
    total_matches = 0
    gt_NO_TR_violation_tracks = {}
    gt_TR_violation_tracks = {}
    all_folders_gt_to_pred = {}
    all_gt_tracks = {}
    all_pred_tracks = {}

    total_pred_riders = 0
    far_riders_discarded_pred = 0
    total_gt_class_tracks = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
    total_gt_NO_TR_violations_tracks = 0
    total_gt_TR_violations_tracks = 0
    total_pred_class_tracks = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
    total_pred_NO_TR_violations_tracks = 0
    total_pred_TR_violations_tracks = 0
    all_count = 0

    import os

    if not os.path.exists("fn_annots"):
        os.makedirs("fn_annots")
    else :
        for file in os.listdir("fn_annots"):
            os.remove(os.path.join("fn_annots", file))

    if not os.path.exists("fp_annots"):
        os.makedirs("fp_annots")
    else :
        for file in os.listdir("fp_annots"):
            os.remove(os.path.join("fp_annots", file))

    if not os.path.exists("tp_annots"):
        os.makedirs("tp_annots")
    else :
        for file in os.listdir("tp_annots"):
            os.remove(os.path.join("tp_annots", file))


    for i, gt_file in enumerate(all_gt_files):
        # if(i >= 30):
        #     continue


        print(gt_file)
        vid_name = gt_file.split("/")[-1].split(".")[0]

        # read the video name from /ssd_scratch/cvit/Video_set_1/videos/vid_name.mp4
        all_frames = []
        vid_path = "/ssd_scratch/cvit/keshav/dashcop/Video_set_1/videos/" + vid_name + ".mp4"
        cap = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        folder = vid_name
        pred_file = all_pred_files[i]
        count = {}

        try:
            with open(pred_file, 'r') as f:
                lines_pred = f.readlines()
        except:
            print("No pred file for ", vid_name)
            continue

        
        # create a folder in challan_rate_calc_input_tr with the name of the folder
        if not os.path.exists('challan_rate_calc_input_tr/' + folder.split('/')[-1]):
            os.makedirs('challan_rate_calc_input_tr/' + folder.split('/')[-1])
        else:
            for filename in os.listdir('challan_rate_calc_input_tr/' + folder.split('/')[-1]):
                os.remove('challan_rate_calc_input_tr/' + folder.split('/')[-1] + '/' + filename)
        challan_matches = []
        challan_fp_background = []
        challan_fn_background = []

        with open(gt_file, 'r') as f:
            lines_gt = f.readlines()

        
        tracks_gt_to_pred = {}
        fp_det_tracks[folder] = []
        fn_det_tracks[folder] = []
        all_gt_tracks[folder] = []
        all_pred_tracks[folder] = []

        our_all_gt_tracks = {}
        our_all_pred_tracks = {}
        our_all_pred_tracks_ids = {}


        gt_NO_TR_violation_track_ids = []
        gt_TR_violation_track_ids = []
        gt_class_track_ids = {0 : [], 1 : [], 2 : [], 3 : []}
        pred_NO_TR_violation_track_ids = []
        pred_TR_violation_track_ids = []
        pred_class_track_ids = {0 : [], 1 : [], 2 : [], 3 : []}

        for frame_num in range(0, 2000, 5):
            our_all_pred_tracks[frame_num] = []
            gt_tracks = []
            pred_tracks = []
            for line in lines_gt:
                line = line.split(' ')
                if int(line[0]) == frame_num:
                    label, xmin, ymin, xmax, ymax, id = int(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), int(line[6])
                    inst_area = (xmax - xmin) * (ymax - ymin)
                    if inst_area/frame_area > inst_area_threshold and label != 0:
                        if(label >= 3):
                            label = 3
                            # print("Triple Riding :", id, frame_num)
                        gt_tracks.append([xmin, ymin, xmax, ymax, id, label])
                        if id not in our_all_gt_tracks:
                            our_all_gt_tracks[id] = []
                        our_all_gt_tracks[id].append([frame_num, xmin, ymin, xmax, ymax, label])
                        gt_class_track_ids[label].append(id)
                        if label != 3:
                            gt_NO_TR_violation_track_ids.append(id)
                        elif label == 3 and id != -1:
                            gt_TR_violation_track_ids.append(id)

            for line in lines_pred:
                line = line.split(' ')
                if int(line[0]) == frame_num:
                    label, xmin, ymin, xmax, ymax, id = int(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), int(line[6])
                    inst_area = (xmax - xmin) * (ymax - ymin)
                    if inst_area/frame_area > inst_area_threshold and label != 0:
                        total_pred_riders += 1
                        our_all_pred_tracks[frame_num].append([xmin, ymin, xmax, ymax, id, label])
                        if id not in our_all_pred_tracks_ids:
                            our_all_pred_tracks_ids[id] = []
                        our_all_pred_tracks_ids[id].append([frame_num, xmin, ymin, xmax, ymax, label])
                        if(label >= 3):
                            label = 3
                        pred_tracks.append([xmin, ymin, xmax, ymax, id, label])
                        pred_class_track_ids[label].append(id)
                        if label != 3:
                            pred_NO_TR_violation_track_ids.append(id)
                        elif label == 3:
                            pred_TR_violation_track_ids.append(id)
                    else:
                        far_riders_discarded_pred += 1
            
            matches, fp, fn = get_matches(gt_tracks, pred_tracks, thresh = 0.3, use_iou = True)
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
                
                if(gt_tracks[match[0]][4] in gt_class_track_ids[3]):
                    if(pred_tracks[match[1]][4] in pred_class_track_ids[3]):
                        # print(pred_tracks[match[1]][4])
                        count[gt_tracks[match[0]][4]] = 1
                        # print("HERE")
            total_matches += len(matches)

            # concatenate the gt_tracks and pred_tracks to all_gt_tracks and all_pred_tracks
            all_gt_tracks[folder] += gt_tracks
            all_pred_tracks[folder] += pred_tracks

        for i in range(4):
            total_gt_class_tracks[i] += len(np.unique(gt_class_track_ids[i]))
            total_pred_class_tracks[i] += len(np.unique(pred_class_track_ids[i]))

        all_matched_pred_track_ids = []
        for k, v in tracks_gt_to_pred.items():
            all_matched_pred_track_ids += v
        
        all_matched_pred_track_ids = np.unique(all_matched_pred_track_ids)
        total_gt_NO_TR_violations_tracks += len(np.unique(gt_NO_TR_violation_track_ids))
        total_gt_TR_violations_tracks += len(np.unique(gt_TR_violation_track_ids))
        total_pred_NO_TR_violations_tracks += len(np.unique(pred_NO_TR_violation_track_ids))
        total_pred_TR_violations_tracks += len(np.unique(pred_TR_violation_track_ids))

        unique_gt_NO_TR_violation_track_ids = np.unique(gt_NO_TR_violation_track_ids)
        matched_gt_NO_TR_violation_track_ids = list(tracks_gt_to_pred.keys())
        false_negatives = np.setdiff1d(unique_gt_NO_TR_violation_track_ids, matched_gt_NO_TR_violation_track_ids)
        for fn in false_negatives:
            challan_fn_background.append([fn, 0])
        
                
        conf_mat[2, 0] += len(false_negatives)

        unique_gt_TR_violation_track_ids = np.unique(gt_TR_violation_track_ids)
        matched_gt_TR_violation_track_ids = list(tracks_gt_to_pred.keys())
        false_negatives = np.setdiff1d(unique_gt_TR_violation_track_ids, matched_gt_TR_violation_track_ids)
        print("TR")
        print(false_negatives)
        print(vid_name)
        for fn in false_negatives:
            gt_id = fn
            gt_frames = our_all_gt_tracks[gt_id]
            for f_idx in gt_frames:
                frame_num = f_idx[0]
                frame = all_frames[frame_num]
                xmin, ymin, xmax, ymax = f_idx[1], f_idx[2], f_idx[3], f_idx[4]
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, str(gt_id), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for f_idx in our_all_pred_tracks.keys():
            if our_all_pred_tracks[f_idx] == []:
                continue
            print(f_idx)
            frame = all_frames[f_idx]
            for pred_track in our_all_pred_tracks[f_idx]:
                xmin, ymin, xmax, ymax = pred_track[0], pred_track[1], pred_track[2], pred_track[3]
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, str(pred_track[4]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for fn in false_negatives:
            for f_idx in our_all_gt_tracks[fn]:
                # save the frame to fn_annots as video_name_frame_num.jpg
                frame_num = f_idx[0]
                frame = all_frames[frame_num]
                cv2.imwrite(f"./fn_annots/{vid_name}_{frame_num}.jpg", frame)
        
        conf_mat[2, 1] += len(false_negatives)
        for fn in false_negatives:
            challan_fn_background.append([fn, 3])

        unique_pred_NO_TR_violations = np.unique(np.unique(pred_NO_TR_violation_track_ids).tolist() + np.unique(pred_TR_violation_track_ids).tolist())
        matched_preds_track_ids = all_matched_pred_track_ids
        false_positives = np.setdiff1d(unique_pred_NO_TR_violations, matched_preds_track_ids)
        folder_pred_tracks = np.array(all_pred_tracks[folder])

        
        for fp in false_positives:
            pred_labels = folder_pred_tracks[folder_pred_tracks[:, -2] == fp][:, -1]
            if(len(pred_labels) <= 1):
                continue
            if(3 in pred_labels):
                conf_mat[1, 2] += 1
            else:
                conf_mat[0, 2] += 1

        unique_pred_TR_violations = np.unique(pred_TR_violation_track_ids)
        matched_preds_track_ids = all_matched_pred_track_ids
        false_positives = np.setdiff1d(unique_pred_TR_violations, matched_preds_track_ids)
        conf_mat[0, 2] += len(false_positives)

        for fp in false_positives:
            fp_id = fp
            pred_frames = our_all_pred_tracks_ids[fp_id]
            for f_idx in pred_frames:
                frame_num = f_idx[0]
                frame = all_frames[frame_num]
                xmin, ymin, xmax, ymax = f_idx[1], f_idx[2], f_idx[3], f_idx[4]
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, str(fp_id), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(f"./fp_annots/{vid_name}_{frame_num}.jpg", frame)

        print("folder: ", folder)
        # print("tracks_gt_to_pred: ", tracks_gt_to_pred)
        all_folders_gt_to_pred[folder] = tracks_gt_to_pred
        # remove the tracks from fp_det_tracks and fn_det_tracks that are in the matches
        for match in matches:
            if pred_tracks[match[1]][4] in fp_det_tracks[folder]:
                fp_det_tracks[folder].remove(pred_tracks[match[1]][4])
            if gt_tracks[match[0]][4] in fn_det_tracks[folder]:
                fn_det_tracks[folder].remove(gt_tracks[match[0]][4])

        all_count += len(count.keys())

    print("ALL COUNT : ", all_count)
    # print("fp_det_tracks: ", fp_det_tracks)
    # print("fn_det_tracks: ", fn_det_tracks)

    total_fp_instances = 0
    total_fn_instances = 0
    for folder in fp_det_tracks:
        total_fp_instances += len(fp_det_tracks[folder])
        total_fn_instances += len(fn_det_tracks[folder])

    print("total_fp_instances: ", total_fp_instances)
    print("total_fn_instances: ", total_fn_instances)
    print("total_matched instances: ", total_matches)

    for i in range(4):
        print(f"total gt {i} instance tracks: ", total_gt_class_tracks[i])
        print(f"total pred {i} instance tracks: ", total_pred_class_tracks[i])

    print("total gt NO_TR_violation rider tracks: ", total_gt_NO_TR_violations_tracks)
    print("total gt TR_violation rider tracks: ", total_gt_TR_violations_tracks)
    print("total pred riders: ", total_pred_riders)
    print("far_riders_discarded_pred: ", far_riders_discarded_pred)
    print("total pred NO_TR_violation rider tracks: ", total_pred_NO_TR_violations_tracks)
    print("total pred no NO_TR_violation rider tracks: ", total_pred_TR_violations_tracks)

    # for every gt box in tracks_gt_to_pred, check the label of the corresponding pred box id that occurs the most number of times in tracks_gt_to_pred[gt_box_id]
    id_switich_count = 0
    correct_instances_NO_TR_violation = 0
    correct_instances_class = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
    total_instances_NO_TR_violation = 0
    correct_instances_TR_violation = 0
    total_instances_TR_violation = 0
    total_instances_class = {0 : 0, 1 : 0, 2 : 0, 3 : 0}

    for vid_name in all_folders_gt_to_pred.keys():
        # if '20211125081955_0060' not in vid_name:
        #     continue
        folder = vid_name
        tracks_gt_to_pred = all_folders_gt_to_pred[folder]
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

            # if(id_switich_count > 0):
                # print("pred_label_counts: ", pred_box_id_counts)

            # print("gt_box_id: ", gt_box_id, "pred_label: ", max_count_pred_label)
            # find a track with gt_box_id in gt_tracks and get label
            gt_label = 0
            for gt_track in all_gt_tracks[folder]:
                if gt_track[4] == gt_box_id:
                    gt_label = gt_track[5]
                    if(gt_label == 3):
                        break
            if(gt_label != 3):
                gt_label = 0
            
            # if(gt_label == 3):
            if(3 in pred_label_counts):
                max_count_pred_label = 3
                # conf_mat[1, 2] += id_switich_count
            else:
                max_count_pred_label = 0
                # conf_mat[0, 2] += id_switich_count
                # print()
                # print(pred_box_ids)
                # print(pred_box_id_counts)
                # print(pred_label_counts)
            

            if gt_label == max_count_pred_label:
                correct_instances_class[gt_label] += 1
                if gt_label != 3:
                    correct_instances_NO_TR_violation += 1
                    conf_mat[0, 0] += 1

                else:
                    correct_instances_TR_violation += 1
                    conf_mat[1, 1] += 1

        
                    print("TR TRUE POSITIVE FOUND", vid_name)
        
            # add in
            else:
                if(gt_label == 0 and max_count_pred_label == 3):
                    conf_mat[1, 0] += 1

                    # print("NON TR")
                    # print(vid_name)
                    # gt_id = gt_box_id
                    # gt_frames = our_all_gt_tracks[gt_id]
                    # for f_idx in gt_frames:
                    #     frame_num = f_idx[0]
                    #     frame = all_frames[frame_num]
                    #     xmin, ymin, xmax, ymax = f_idx[1], f_idx[2], f_idx[3], f_idx[4]
                    #     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    #     cv2.putText(frame, str(gt_id), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # for f_idx in our_all_pred_tracks.keys():
                    #     if our_all_pred_tracks[f_idx] == []:
                    #         continue
                    #     frame = all_frames[f_idx]
                    #     for pred_track in our_all_pred_tracks[f_idx]:
                    #         xmin, ymin, xmax, ymax = pred_track[0], pred_track[1], pred_track[2], pred_track[3]
                    #         cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    #         cv2.putText(frame, str(pred_track[4]), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # for f_idx in our_all_gt_tracks[gt_id]:
                    #     # save the frame to fn_annots as video_name_frame_num.jpg
                    #     frame_num = f_idx[0]
                    #     frame = all_frames[frame_num]
                    #     cv2.imwrite(f"./fn_annots/{vid_name}_{frame_num}.jpg", frame)


                elif(gt_label == 3 and max_count_pred_label == 0):
                    conf_mat[0, 1] += 1

            string_pred_box_ids = ''
            for pred_box_id in np.unique(pred_box_ids):
                string_pred_box_ids += str(pred_box_id) + ','
            string_pred_box_ids = string_pred_box_ids[:-1]
            challan_matches.append([gt_box_id, gt_label, max_count_pred_label,string_pred_box_ids])

            total_instances_class[gt_label] += 1
                
            if gt_label != 3:
                total_instances_NO_TR_violation += 1
            else:
                total_instances_TR_violation += 1

        

    print("id_switich_count: ", id_switich_count)

    for i in range(0, 4):
        print(f"correct_instances {i}: ", correct_instances_class[i])
        print(f"total instances {i}: ", total_instances_class[i])

    print("correct_instances_NO_TR_violation: ", correct_instances_NO_TR_violation)
    print("total_instances_NO_TR_violation: ", total_instances_NO_TR_violation)
    print("correct_instances_TR_violation: ", correct_instances_TR_violation)
    print("total_instances_TR_violation: ", total_instances_TR_violation)

    print(conf_mat)

    # normalize the confmat across columns
    normalized_conf_mat = np.zeros((3, 3))
    for i in range(3):
        normalized_conf_mat[:,i] = conf_mat[:,i]/np.sum(conf_mat[:,i])

    print(normalized_conf_mat)
    tp = normalized_conf_mat[1, 1]
    fp = normalized_conf_mat[1, 0] + normalized_conf_mat[1, 2]
    fn = normalized_conf_mat[0, 1] + normalized_conf_mat[2, 1]

    print("Precision: ", tp/(tp + fp))
    print("Recall: ", tp/(tp + fn))
    print("F1 Score: ", 2*tp/(2*tp + fp + fn))
    max_f1_score = max(max_f1_score, 2*tp/(2*tp + fp + fn))
    if(max_f1_score == 2*tp/(2*tp + fp + fn)):
        max_thresh = inst_area_threshold

print("Max F1 Score: ", max_f1_score)
print("Max Threshold: ", max_thresh)
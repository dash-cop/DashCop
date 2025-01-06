import sys
sys.path.append('../')
sys.path.append('tracker')
sys.path.append('/home2/keshav06/.local/lib/python3.6/site-packages')
import os
import shutil
import time
from absl import app, flags, logging
from absl.flags import FLAGS
# from core.association import *
# from core.config import cfg
from PIL import Image
import cv2
import numpy as np
# import pickle
import torch
import matplotlib.pyplot as plt
from instance_funcs import *

from ultralytics import YOLO
# from core.association import *
import matplotlib.pyplot as plt

from transformers import VisionEncoderDecoderModel, TrOCRForCausalLM
from transformers import TrOCRProcessor
from pipeline_comp.det_assoc import DetAssoc
from pipeline_comp.yolo_clf import YOLOClf
# from pipeline_comp.lp import LicensePlateModel
from pipeline_comp.hnh import HNHModel

import sys
sys.path.append('../')
sys.path.append('tracker')
sys.path.append('tracker/fast_reid')
sys.path.append('tracker/fast_reid/fastreid')
sys.path.append('tracker/fast_reid/fastreid/data')
from tracker.assoc_tracker.assoc_tracker import *

flags.DEFINE_string('video', '/ssd_scratch//cvit/keshav/dashcop/Video_set_1/videos/20211109152409_0060.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '/ssd_scratch/cvit/keshav/outputs/20211109152409_0060_inferred.mp4', 'path to raw output video')
flags.DEFINE_integer('frame_start', 0, 'frame start')
flags.DEFINE_integer('frame_max', 100, 'frame max')
flags.DEFINE_boolean('infer_lp', False, 'infer license plates or not')
flags.DEFINE_boolean('infer_hnh', True, 'infer helmet/no-helmet or not')
flags.DEFINE_boolean('clf', True, 'use clf or not')
flags.DEFINE_string('tracker_config', './tracker/assoc_tracker/base_cfg.yaml', 'path to assoc tracker config')
flags.DEFINE_string('rm_weights', '/ssd_scratch/cvit/keshav/weights/model_ft.pt', 'path to rider motorcyle assoc weights')
flags.DEFINE_string('hnh_weights', '/ssd_scratch/cvit/keshav/weights/hnh_frame.pt', 'path to helmet/no-helmet weights')
flags.DEFINE_string('clf_weights', '/ssd_scratch/cvit/keshav/weights/tr_clf.ckpt', 'path to helmet/no-helmet weights')
flags.DEFINE_string('lp_det_weights', '/ssd_scratch/cvit/keshav/weights/lp_det.pt', 'path to lp det weights')
flags.DEFINE_string('lp_rec_weights', '/ssd_scratch/cvit/keshav/weights/lp_rec', 'path to lp-rec model -> dtrb weights')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')

def main(_argv):
    
    tracker_assoc = AssocTracker(FLAGS.tracker_config)
    rm_model = DetAssoc(weights_path=FLAGS.rm_weights, conf_score=0.5)
    clf = FLAGS.clf
    save_root = "/ssd_scratch/cvit/keshav/inferred_vids_assoc/" # add a / in the end
    txt_file_root = "/ssd_scratch/cvit/keshav/txt_files_assoc/" # add a / in the end
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(txt_file_root, exist_ok=True)

    if(not clf):
        FLAGS.output = save_root + FLAGS.video.split("/")[-1][:-4] + "_det.mp4"
    else:
        FLAGS.output = save_root + FLAGS.video.split("/")[-1][:-4] + ".mp4"
        dt_classifier = YOLOClf(weights_path=FLAGS.clf_weights, rm_assoc_path=FLAGS.rm_weights)

    if(FLAGS.infer_lp):
        lp_model = LicensePlateModel(det_weights=FLAGS.lp_det_weights, rec_weights=FLAGS.lp_rec_weights)
        all_frame_lp_boxes, all_frame_lp_numbers = [], []
    if(FLAGS.infer_hnh):
        hnh_model = HNHModel(weights_path=FLAGS.hnh_weights)
        all_frame_hnh_boxes, all_frame_hnh_classes = [], []

    # begin video capture
    video_path = FLAGS.video
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    frame_num = 0
    frame_start = FLAGS.frame_start #500
    frame_max = FLAGS.frame_max #600
    all_frames_imgs = []
    all_masks = []

    t0 = time.time()
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break

        if(frame_num == frame_max):
            break

        if(frame_num < frame_start):
            frame_num += 1
            continue
        all_frames_imgs.append(frame)
        print('\n\nFrame #: ', frame_num)
        
        frame_num += 1

        # all_preds_data, all_masks, all_cross_masks = rm_model(frame)
        all_preds_data, all_masks, all_cross_masks = rm_model(frame)
        det_feats = tracker_assoc.track(all_preds_data, all_masks, all_cross_masks, frame)

        if FLAGS.infer_hnh:
            hnh_bboxes, _, hnh_class = hnh_model(frame)
            all_frame_hnh_boxes.append(hnh_bboxes)
            all_frame_hnh_classes.append(hnh_class)
        
        if FLAGS.infer_lp:
            lp_bboxes, lp_nums = lp_model(frame)
            all_frame_lp_boxes.append(lp_bboxes)
            all_frame_lp_numbers.append(lp_nums)
    

    tracks_rider, tracks_motor, motor_rider_count = tracker_assoc.give_tracks()
    # print(tracks_rider)
    # print(tracks_motor)

    all_mot_data_rider = []
    all_mot_data_motor = []
    all_data = []
    all_data.append(str(all_frames_imgs[0].shape[:2][::-1]))
    all_mot_data_motor.append(str(all_frames_imgs[0].shape[:2][::-1]))
    all_mot_data_rider.append(str(all_frames_imgs[0].shape[:2][::-1]))

    for frame_num in range(len(all_frames_imgs)):
        # try:
        img = all_frames_imgs[frame_num]
        cv2.putText(img, f"{frame_num}", (40, 40), 2, 2, (0, 255, 0), 2)

        r_track = tracks_rider[frame_num + 1]
        m_track = tracks_motor[frame_num + 1]
        if(FLAGS.infer_hnh):
            hnh_boxes = all_frame_hnh_boxes[frame_num]
            hnh_classes = all_frame_hnh_classes[frame_num]
            for i, box in enumerate(hnh_boxes):
                l, t, r, b = box.astype(np.int32)
                cls = hnh_classes[i]
                if(cls == 0):
                    cv2.rectangle(img, (l, t), (r, b), (255, 255, 255), 1)
                else:
                    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)

        assocs = {}

        for tr in r_track:
            id, bb, aid = tr
            bb_np = np.array(bb)[:4]
            if(len(bb_np[bb_np <= 0]) == 4):
                continue
            if(aid != -1):
                try:
                    assocs[aid].append((id, bb, 0))
                except:
                    assocs[aid] = [(id, bb, 0)]
            l, t, r, b, _ = bb
            if(frame_num % 5 == 0):
                line = f"{frame_num + frame_start} {int(id) + 1} {l} {t} {r} {b} -1 1 -1"
                r_line = f"{frame_num + frame_start} {0} {l} {t} {r} {b} {id} {aid}"
                all_data.append(r_line)
            #     # all_hnh_data.append(h_line)
                all_mot_data_rider.append(line)

            cv2.rectangle(img, (l, t), (r, b), (255, 255, 255), 1)
            cv2.putText(img, f"{id}", (l, t), 2, 1, (255, 255, 255), 2)
            # cv2.putText(img, f"{aid}", (l + 20, t + 20), 2, 2, (0, 255, 0), 2)
        
        for tr in m_track:
            id, bb, aid = tr
            bb_np = np.array(bb)[:4]
            if(len(bb_np[bb_np <= 0]) == 4):
                continue
            try:
                assocs[aid].append((id, bb, 1))
            except:
                assocs[aid] = [(id, bb, 1)]
            l, t, r, b, _ = bb
            if(frame_num % 5 == 0):
                line = f"{frame_num + frame_start} {int(id) + 1} {l} {t} {r} {b} -1 1 -1"
                m_line = f"{frame_num + frame_start} {1} {l} {t} {r} {b} {id} {aid}"
                all_data.append(m_line)
                all_mot_data_motor.append(line)
            cv2.rectangle(img, (l, t), (r, b), (255, 255, 0), 1)
            cv2.putText(img, f"{id}", (l, t), 2, 1, (255, 0, 255), 2)
            # cv2.putText(img, f"{aid}", (l + 20, t + 20), 2, 2, (255, 255, 255), 2)
        
        if(FLAGS.infer_lp):
            lp_boxes = all_frame_lp_boxes[frame_num]
            lp_nums = all_frame_lp_numbers[frame_num]
            for i, box in enumerate(lp_boxes):
                l, t, r, b = box.astype(np.int32)
                num = lp_nums[i]
                max_iou = 0
                mtr_idx = -1
                for i, tr in enumerate(m_track):
                    id, bb, aid = tr
                    bb_np = np.array(bb)[:4]
                    if(len(bb_np[bb_np <= 0]) == 4):
                        continue
                    iou = calculate_iou(bb_np, box.astype(np.int32))
                    if(iou > max_iou):
                        max_iou = iou
                        mtr_idx = i
                if(max_iou):
                    cv2.rectangle(img, (l, t), (r, b), (255, 255, 255), 1)
                    cv2.putText(img, f"{num}", (l, t), 1, 1, (255, 255, 0), 2)
        
        # if(FLAGS.infer_hnh):
        #     for i, hnh in enumerate(hnh_boxes):
        #         l, t, r, b = hnh
        #         class_ = hnh_classes[i] + 2
        #         hnh_line = f"{frame_num + frame_start} {class_} {l} {t} {r} {b} {-1} {-1}"
        #         all_data.append(hnh_line)

        if(FLAGS.infer_lp):
            for i, lp in enumerate(lp_boxes):
                 l, t, r, b = lp
                 num = lp_nums[i]
                 lp_line = f"{frame_num + frame_start} {4} {l} {t} {r} {b} {num} {-1}"
                 all_data.append(lp_line)
        
        all_instances_boxes = {}
        
        for aid, instance in assocs.items():
            xmin, ymin, xmax, ymax = img.shape[1], img.shape[0], -1, -1
            for inst in instance:
                id, bb, cls = inst
                bb_np = np.array(bb)[:4]
                if(len(bb_np[bb_np <= 0]) == 4):
                    continue
                l, t, r, b, _ = bb
                xmin = min(xmin, l)
                xmax = max(xmax, r)
                ymin = min(ymin, t)
                ymax = max(ymax, b)
            l, t, r, b = xmin, ymin, xmax, ymax
            all_instances_boxes[aid] = [l, t, r, b]
            
        if(FLAGS.infer_hnh):
            hnh_boxes = all_frame_hnh_boxes[frame_num]
            hnh_mapping = {}
            aid_hnh_mapping = {}
            for i, hnh_box in enumerate(hnh_boxes):
                max_iou, max_iou_aid = 0, -1
                for aid, instance in assocs.items():
                    inst_box = all_instances_boxes[aid]
                    iou = calculate_iou(hnh_box, inst_box)
                    if(iou > max_iou):
                        max_iou = iou
                        max_iou_aid = aid
                hnh_mapping[i] = aid
                try:
                    aid_hnh_mapping[aid].append(i)
                except:
                    aid_hnh_mapping[aid] = [i]
        
        for aid, instance in assocs.items():
            l, t, r, b = all_instances_boxes[aid]
            try:
                inst_crop = img[t:b, l:r, :]
                inst_crop = cv2.cvtColor(inst_crop, cv2.COLOR_BGR2RGB) 
                output = dt_classifier(inst_crop)
                num_riders = output + 1
            except:
                num_riders = 0
            
            hnh_inst_violation = False
            if(FLAGS.infer_hnh):
                if(aid in aid_hnh_mapping):
                    hnh_indices = aid_hnh_mapping[aid]
                    for hnh_ind in hnh_indices:
                        curr_cls = hnh_classes[hnh_ind]
                        if(curr_cls == 0):
                            hnh_inst_violation = True
                            break
            
            if(not clf):
                num_riders = len(instance) - 1 # 1 motorcycle, rest riders
            
            if(num_riders >= 3):
                cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 3)
                cv2.putText(img, f"Triple Rider Violation", (l - 10, t + 20), 2, 1.2, (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
                
            if(hnh_inst_violation):
                cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 3)
                cv2.putText(img, f"Helmet Violation", (l - 10, t + 20), 2, 1.2, (0, 0, 255), 2)
            
            # if(motor_rider_count[aid] >= 3):
            #     cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 3)
            # else:
            #     cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)

            # cv2.putText(img, f"Track ID:{aid}", (l - 10, t + 20), 2, 1.2, (255, 255, 0), 2)
            # cv2.putText(img, f"DTC:{output[0]}", (l - 10, t + 30), 2, 1.2, (255, 255, 0), 2)

        if(not FLAGS.dont_show):
            cv2.imshow("tracks", img)
            cv2.waitKey(0)

    mot_file_name = FLAGS.video
    all_data_file_name = txt_file_root + mot_file_name.split('/')[-1].split('.')[0] + "_data.txt"
    mot_file_name_rider = txt_file_root + mot_file_name.split('/')[-1].split('.')[0] + f'_{frame_start}_{frame_max}_rider.txt'
    mot_file_name_motor = txt_file_root + mot_file_name.split('/')[-1].split('.')[0] + f'_{frame_start}_{frame_max}_motor.txt'

    with open(mot_file_name_rider, "w") as f:
        for line in all_mot_data_rider:
            f.write(line + "\n")

    with open(mot_file_name_motor, "w") as f:
        for line in all_mot_data_motor:
            f.write(line + "\n")

    with open(all_data_file_name, "w") as f:
        for line in all_data:
            f.write(line + "\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = all_frames_imgs[0].shape
    out = cv2.VideoWriter(FLAGS.output, fourcc, 30.0, (width, height))
    for frame in all_frames_imgs:
        out.write(frame)
    out.release()

    print("LOOP 1 OVER")
    print("##############")
    print((time.time() - t0) / 60)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
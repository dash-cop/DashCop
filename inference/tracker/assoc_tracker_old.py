import torch
from .trackers import AssocTracker as AssociationTracker
import yaml
import re
from munch import DefaultMunch
import numpy as np

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

class Detections():
    def __init__(self, preds, masks, cross_masks):
        self.preds = preds
        self.masks = masks
        self.cross_masks = cross_masks

class AssocTracker():
    def __init__(self, cfg_file):
        cfg = yaml_load(cfg_file)
        cfg = DefaultMunch.fromDict(cfg)
        self.tracker = AssociationTracker(args=cfg, frame_rate=30)

    def update_(self, det_riders, det_motors, img):
        '''
        det is an object with parameters
            conf : numpy array of shape(N) where each element corresponds to the confidence of the detection
            cls : numpy array of shape(N) where each element corresponds to the class-id of the detection
            xyxy : numpy array of shape(N,4) where each element corresponds to the xyxy coordinates of the detection
        img is a numpy array of shape (H,W,3), the frame image itself

        Returns the list of the tracks in the form of a numpy array of shape (M, 8)
        where M are the number of active tracks, and each row is of the form
        xyxy(tlbr) , track_id, score, class, idx in the det array with N detections
        '''

        tracks_rider, tracks_motor, track_assoc_dict = self.tracker.update(det_riders, det_motors, img)
        return (tracks_rider, tracks_motor, track_assoc_dict)
    
    def update(self, all_preds_data, all_masks, all_cross_masks, img):
        all_classes = all_preds_data[:, 5]

        rider = all_classes == 0
        motor = all_classes == 1

        rider_preds = all_preds_data[rider]
        motor_preds = all_preds_data[motor]
        rider_masks = all_masks[rider]
        motor_masks = all_masks[motor]
        rider_cross_masks = all_cross_masks[rider]
        motor_cross_masks = all_cross_masks[motor]

        tracks_rider, tracks_motor, track_assoc_dict = self.update_(Detections(rider_preds, rider_masks, rider_cross_masks),
                                          Detections(motor_preds, motor_masks, motor_cross_masks), img)

        # if(len(tracks)!=0):
        #     track_bbox = tracks[:, :4]
        #     track_ids = tracks[:, 4]
        #     track_cls = tracks[:, 6]
        #     track_idx = tracks[:, 7]
        # else:
        #     track_bbox = np.zeros(0)
        #     track_ids = np.zeros(0)
        #     track_cls = np.zeros(0)
        #     track_idx = np.zeros(0)
        # return track_bbox, track_ids, track_cls, track_idx

        return tracks_rider, tracks_motor, track_assoc_dict


    def predict(self):
        # Update function performs both predict and update simultaneously
        pass
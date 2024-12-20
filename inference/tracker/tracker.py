import torch
from .trackers import BOTSORT, BYTETracker
import yaml
import re
from munch import DefaultMunch
import numpy as np

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

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
    def __init__(self, bbox, scores, cls_id):
        self.xyxy = bbox
        self.conf = scores
        self.cls = cls_id


class Tracker():
    def __init__(self, tracker_algo, cfg_file):
        cfg = yaml_load(cfg_file)
        cfg = DefaultMunch.fromDict(cfg)
        self.tracker = TRACKER_MAP[tracker_algo](args=cfg, frame_rate=30)

    def update_(self, det, img):
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

        tracks, keypoints = self.tracker.update(det, img)
        # print(tracks)
        return (tracks, keypoints)
    
    def update(self, bbox, score, cls_id, img):
        # bbox in tlbr
        tracks, keypoints = self.update_(Detections(bbox, score, cls_id), img)
        # print('TRACKS', tracks)
        if(len(tracks)!=0):
            track_bbox = tracks[:, :4]
            track_ids = tracks[:, 4]
            track_cls = tracks[:, 6]
            track_idx = tracks[:, 7]
        else:
            track_bbox = np.zeros(0)
            track_ids = np.zeros(0)
            track_cls = np.zeros(0)
            track_idx = np.zeros(0)
        return track_bbox, track_ids, track_cls, track_idx


    def predict(self):
        # Update function performs both predict and update simultaneously
        pass
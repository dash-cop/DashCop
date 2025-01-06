import numpy as np
from utils.kalman_filter import KalmanFilterXYWH
from fast_reid.fast_reid_interfece import FastReIDInterface
import torch
import copy
import gurobipy as gp
from scipy.interpolate import interp1d
from pulp import *
import time
from tracker.assoc_tracker.tracker_dists import TrackerDists
from .node import Node
from .detection import Detection
import json
import yaml
from .tracker import Tracker


# REID_MAPPING_FILE = "/Users/keshavgupta/desktop/CVIT/TrafficViolations/simil_off_diag.npy"

# TODO
# Not penalizing -1

# TODO : Can make this MUCH MORE memory efficent as Node contains detections and we are making many copies of a given node.
# We can avoid that by having a global memory for all the node.detections

class AssocTracker():
    def __init__(self, cfg, reid_model=None):
        if(type(cfg) == str):
            if(cfg.endswith("json")):
                cfg = self.read_json_cfg(cfg)
            elif(cfg.endswith("yaml")):
                cfg = self.read_yaml_cfg(cfg)

        self.cfg = cfg
        self.logger = self.null
        self.frame_num = 0
        self.prune_after = 1
        self.eps = 0

        self.rider_obj_weight = self.cfg.get('rider_obj_weight', 1)
        self.motor_obj_weight = self.cfg.get('motor_obj_weight', 1)
        self.assoc_weight = self.cfg.get('assoc_weight', 1)
        self.assoc_thresh = self.cfg.get('assoc_thresh', 0.4)
        self.last_n_frames = self.cfg.get("last_n_frames", 40)
        self.max_miss_count = self.cfg.get("max_miss_count", 20)
        self.max_tree_depth = self.cfg.get("max_tree_depth", 25)
        assert self.max_tree_depth > self.max_miss_count, "Max Tree Depth should be more than the max miss count"
        print("TrHERE")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if(reid_model is None):
            self.reid_encoder = FastReIDInterface(config_file=self.cfg.get("REID_CONFIG"), weights_path=self.cfg.get("REID_WEIGHTS"), device=self.device)
        else:
            self.reid_encoder = reid_model
        print("REID LOADED")
        self.rider_tracker = Tracker("rider", self.prune_after, self.logger, self.reid_encoder, self.max_tree_depth, cfg)
        self.motor_tracker = Tracker("motor", self.prune_after, self.logger, self.reid_encoder, self.max_tree_depth, cfg)
        
        self.masks_riders = None
        self.cross_masks_riders = None
        self.masks_motors = None
        self.cross_masks_motors = None
        self.boxes_riders = None
        self.boxes_motors = None
        self.mask_shape = [50, 100]
        self.mask_shape = [93, 160]

        self.curr_rider_tracks = {}
        self.curr_motor_tracks = {}
        self.curr_assocs = {} # motor to rider
        self.curr_assocs_inv = {} # rider to motor

        self.make_problem_time = []

    def read_json_cfg(self, cfg):
        with open(cfg, 'r') as f:
            contents = json.load(f)
        return contents

    def read_yaml_cfg(self, cfg):
        with open(cfg, 'r') as f:
            contents = yaml.load(f, Loader=yaml.loader.SafeLoader)
        return contents

    def append_to_array(self, arr, elem):
        num_dets = elem.shape[0]
        max_dets_till_now = arr.shape[1] if arr is not None else 0
        new_max_dets = max(num_dets, max_dets_till_now)
        print("*****************")
        print("MAX_DETS_TILL_NOW : ", new_max_dets)   
        print("*****************")
        if(len(elem.shape) == 1):
            # if elem is []
            elem = np.zeros((len(elem), *self.mask_shape)).astype(np.uint8)
        if(arr is not None):
            start = 0
            new_arr_size = arr.shape[0] + 1
            if(self.frame_num > self.last_n_frames):
                new_arr_size = self.last_n_frames
                start = 1
            new = np.zeros((new_arr_size, new_max_dets, *self.mask_shape)).astype(np.uint8)
            new[:-1, :max_dets_till_now] = arr[start:]
            new[-1, :elem.shape[0]] = elem
        else:
            new = elem[None]
        return new

    def append_box_to_array(self, arr, elem):
        # arr is (t, max_dets, 4)
        num_dets = elem.shape[0]
        max_dets_till_now = arr.shape[1] if arr is not None else 0
        new_max_dets = max(num_dets, max_dets_till_now)
        print("*****************")
        print("MAX_DETS_TILL_NOW : ", new_max_dets)   
        print("*****************")
        if(len(elem.shape) == 1):
            # if elem is []
            elem = np.zeros((len(elem), 4))
        if(arr is not None):
            start = 0
            new_arr_size = arr.shape[0] + 1
            if(self.frame_num > self.last_n_frames):
                new_arr_size = self.last_n_frames
                start = 1
            new = np.zeros((new_arr_size, new_max_dets, 4))
            new[:-1, :max_dets_till_now] = arr[start:]
            new[-1, :elem.shape[0]] = elem
        else:
            new = elem[None]
        return new
    
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def track(self, detections, masks, cross_masks, frame, rider_reid_feats=None, motor_reid_feats=None):
        '''
        detections are the new detections in the frame. Assuming a new scan from the sensor. It is a numpy array of shape (N, 6)
        representing 
            the coordinates of the bounding box detection in (xyxy format),
            the confidence of the detection,
            the class of the detection (0 for rider, 1 for motor)
        
        masks is the (N, H, W) array holding the masks of the correponding detections
        cross_masks is the (N, H, W) array holding the cross masks of the corresponding detections

        frame is a numpy array of shape (H, W, 3) representing the image of the environment

        The function updates the Tracker with the new detections
        '''
        # self.last_n_frames = self.frame_num
        self.frame_num += 1
        if(len(detections) == 0):
            rider_dets = []
            motor_dets = []
        else:
            cls = detections[:, -1]
            rider_dets = detections[cls == 0][:, :-1]
            rider_boxes = detections[cls == 0][:, :-2]
            motor_dets = detections[cls == 1][:, :-1]
            motor_boxes = detections[cls == 1][:, :-2]
        
        if(self.frame_num % self.prune_after == 0):
            # make gp model
            model = gp.Model()
        else:
            model = None
        print("LEN", len(rider_dets), len(motor_dets))
        
        rider_feats, rider_vars, rider_all_paths, rider_obj, rider_costs = self.rider_tracker.track(rider_dets, frame, model, reid_features=rider_reid_feats)
        motor_feats, motor_vars, motor_all_paths, motor_obj, motor_costs = self.motor_tracker.track(motor_dets, frame, model, reid_features=motor_reid_feats)

        # Need for defining the association costs while making the problem
        if(len(detections) != 0):
            self.masks_riders = self.append_to_array(self.masks_riders, masks[cls == 0].astype(np.uint8))
            self.masks_motors = self.append_to_array(self.masks_motors, masks[cls == 1].astype(np.uint8))
            self.cross_masks_riders = self.append_to_array(self.cross_masks_riders, cross_masks[cls == 0].astype(np.uint8))
            self.cross_masks_motors = self.append_to_array(self.cross_masks_motors, cross_masks[cls == 1].astype(np.uint8))
            self.boxes_riders = self.append_box_to_array(self.boxes_riders, rider_boxes)
            self.boxes_motors = self.append_box_to_array(self.boxes_motors, motor_boxes)
        else:
            self.masks_riders = self.append_to_array(self.masks_riders, np.array([]))
            self.masks_motors = self.append_to_array(self.masks_motors, np.array([]))
            self.cross_masks_riders = self.append_to_array(self.cross_masks_riders, np.array([]))
            self.cross_masks_motors = self.append_to_array(self.cross_masks_motors, np.array([]))
            self.boxes_riders = self.append_box_to_array(self.boxes_riders, np.array([]))
            self.boxes_motors = self.append_box_to_array(self.boxes_motors, np.array([]))


        if(self.mask_shape is None):
            self.mask_shape = masks[0].shape

        if(self.frame_num % self.prune_after == 0):
            self.n_scan_pruning(model, rider_vars, motor_vars, rider_all_paths, motor_all_paths, rider_obj, motor_obj, rider_costs, motor_costs)

        return rider_feats, motor_feats
    
    def calc_assoc_score(self, rider, motor):
        
        # def get_property(obj, attr):
        #     return getattr(obj, attr)

        # get_property_vectorized = np.vectorize(get_property)

        # rider_det_ids = np.array(get_property_vectorized(rider, "det_id"))
        # rider_frame_num = np.array(get_property_vectorized(rider, "frame_num")) - 1
        # motor_det_ids = np.array(get_property_vectorized(motor, "det_id"))
        # motor_frame_num = np.array(get_property_vectorized(motor, "frame_num")) - 1
        # rider_dets_ = np.ones(self.frame_num, dtype=np.int32) * 1
        # rider_dets_[rider_frame_num[0]:] = rider_det_ids
        # motor_dets_ = np.ones(self.frame_num, dtype=np.int32) * 1
        # motor_dets_[motor_frame_num[0]:] = motor_det_ids

        # curr_motor_masks = self.masks_motors[np.arange(self.frame_num, dtype=np.int32), motor_dets_]
        # curr_motor_masks[motor_dets_ == -1] = np.zeros(self.mask_shape)
        
        # curr_cross_rider_masks = self.cross_masks_riders[np.arange(self.frame_num, dtype=np.int32), rider_dets_]
        # curr_cross_rider_masks[rider_dets_ == -1] = np.zeros(self.mask_shape)

        # # Compute iou
        # score = np.zeros(self.frame_num)
        # intersection = (curr_cross_rider_masks * curr_motor_masks).reshape(self.frame_num, -1).sum(-1)
        # union = (curr_cross_rider_masks + curr_motor_masks).reshape(self.frame_num, -1)
        # union[union > 1] = 1
        # union = np.sum(union, -1)
        # cond = (motor_dets_ != -1) * (rider_dets_ != -1)
        # score[cond] = intersection[cond] / union[cond] - 0.5
        # cond = (motor_dets_ != -1) | (rider_dets_ == 1)
        # score[cond] = 0.5 - self.sigmoid((curr_cross_rider_masks[cond]).reshape(-1, self.mask_shape[0]*self.mask_shape[1]).sum(-1))
        # score = self.sigmoid(score.sum())
        # return score
        
        rider_dets = [None] * self.frame_num
        motor_dets = [None] * self.frame_num
        for det in rider:
            # if(det.det_id != -1):
                rider_dets[det.frame_num - 1] = det
        for det in motor:
            # if(det.det_id != -1):
                motor_dets[det.frame_num - 1] = det

        # print("Rider Dets : ", rider_dets)
        # print("Motor Dets", motor_dets)
        score = 0
        for i in range(self.frame_num):
            rider_det = rider_dets[i]
            motor_det = motor_dets[i]
            if(rider_det is None or motor_det is None):
                continue
            
            rider_det_id = rider_det.det_id
            motor_det_id = motor_det.det_id

            if(motor_det_id == -1 and rider_det_id == -1):
                continue
            mask_rider = self.masks_riders[i][rider_det_id]
            mask_motor = self.masks_motors[i][motor_det_id]
            if(rider_det_id == -1):
                cross_mask_rider = np.zeros_like(mask_motor)
            else:
                cross_mask_rider = self.cross_masks_riders[i][rider_det.det_id]
            if(motor_det_id == -1):
                mask_motor = np.zeros_like(mask_rider)
            else:
                mask_motor = self.masks_motors[i][motor_det.det_id]
            # cross_mask_motor = self.cross_masks_motors[i][motor_det.det_id]

            # Calculate the iou between the mask_motor and the cross_mask_rider
            if(motor_det_id == -1):
                # The predicted motor mask should be a null mask
                val = 0.5 - 1 / (1 + np.exp(-np.sum(cross_mask_rider)))
                score += val
                # print(np.sum(cross_mask_rider), val)
            else:
                intersection = np.sum(mask_motor * cross_mask_rider)
                union = mask_motor + cross_mask_rider
                union[union >= 1] = 1
                union = np.sum(union)
                iou = intersection / union
                score += (iou - 0.5)
                # print(i, iou)
        
        score = 1 / (1 + np.exp(-score))
        # print("Score : ", score)
        return score

    def make_problem_identity(self, model, rider_vars, rider_all_paths, rider_obj, rider_costs, motor_vars, motor_all_paths, motor_obj, motor_costs):
        # Add len(tracks_rider) * len(motor_vars) constraints to the problem
        assoc_variables = []
        for i in range(len(rider_all_paths)):
            rider_assocs = []
            rider_aux_var = []
            for j in range(len(motor_vars)):
                rider_assocs.append(model.addVar(vtype=gp.GRB.BINARY, name=f"assoc_{i}_{j}"))
                rider_aux_var.append(model.addVar(vtype=gp.GRB.BINARY, name=f"aux_{i}_{j}"))
            assoc_variables.append(rider_assocs)
        return assoc_variables

    def make_scores(self, assoc_scores_mcr, assoc_scores_rcm, frame_count, tracks_rider_dets, tracks_motor_dets, rider_all_paths, motor_all_paths, tracks_rider_tids, tracks_motor_tids):
        # The function sets the assoc_scores_mcr and the assoc_scores_rcm variables
        # self.masks_motors is of shape (T, M, H, W) and tracks_motor_dets is of shape (N, T)
        curr_motor_masks = []
        start = 0
        for i in range(len(motor_all_paths)):
            to_append = self.masks_motors[np.arange(start, start + frame_count, dtype=np.int32), tracks_motor_dets[i]]
            to_append[tracks_motor_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_motor_masks.append(to_append)
        curr_motor_masks = np.array(curr_motor_masks)

        curr_cross_rider_masks = []
        for i in range(len(rider_all_paths)):
            to_append = self.cross_masks_riders[np.arange(start, start + frame_count, dtype=np.int32), tracks_rider_dets[i]]
            to_append[tracks_rider_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_cross_rider_masks.append(to_append)
        curr_cross_rider_masks = np.array(curr_cross_rider_masks)

        curr_rider_masks = []
        for i in range(len(rider_all_paths)):
            to_append = self.masks_riders[np.arange(start, start + frame_count, dtype=np.int32), tracks_rider_dets[i]]
            to_append[tracks_rider_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_rider_masks.append(to_append)
        curr_rider_masks = np.array(curr_rider_masks)

        curr_cross_motor_masks = []
        for i in range(len(motor_all_paths)):
            to_append = self.cross_masks_motors[np.arange(start, start + frame_count, dtype=np.int32), tracks_motor_dets[i]]
            to_append[tracks_motor_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_cross_motor_masks.append(to_append)
        curr_cross_motor_masks = np.array(curr_cross_motor_masks)

        for i in range(len(curr_motor_masks)):
            scores = np.zeros_like(tracks_rider_dets).astype(np.float32) + self.eps
            intersection = (curr_cross_rider_masks & curr_motor_masks[i][None]).reshape(len(rider_all_paths), frame_count, -1).sum(-1)
            union = (curr_cross_rider_masks | curr_motor_masks[i][None]) * (curr_motor_masks[i][None] > 0)
            union = union.reshape(len(rider_all_paths), frame_count, -1).sum(-1)
            cond = (tracks_motor_dets[i][None] != -1) * (tracks_rider_dets != -1)
            scores[cond] = np.float32(intersection[cond]) / np.float32(union[cond]) - self.assoc_thresh
            scores[np.isnan(scores)] = 0
            scores[np.isinf(scores)] = 0
            assoc_scores_mcr[:, i] = np.sum(scores, -1)

            # scores[scores < 0] = -1000

        for i in range(len(curr_rider_masks)):
            scores = np.zeros_like(tracks_motor_dets).astype(np.float32) + self.eps
            intersection = (curr_cross_motor_masks & curr_rider_masks[i][None]).reshape(len(motor_all_paths), frame_count, -1).sum(-1)
            union = (curr_cross_motor_masks | curr_rider_masks[i][None]) * (curr_rider_masks[i][None] > 0)
            union = union.reshape(len(motor_all_paths), frame_count, -1).sum(-1)
            cond = (tracks_motor_dets != -1) * (tracks_rider_dets[i][None] != -1)
            scores[cond] = np.float32(intersection[cond]) / np.float32(union[cond]) - self.assoc_thresh
            scores[np.isnan(scores)] = 0
            scores[np.isinf(scores)] = 0
            scores[scores < 0] = -1000

            assoc_scores_rcm[i, :] = np.sum(scores, -1)

            curr_rider_id = tracks_rider_tids[i]

            if(curr_rider_id in self.curr_assocs_inv):
                assoc_motor_id = self.curr_assocs_inv[curr_rider_id]
                in_valid_paths = tracks_motor_tids != assoc_motor_id
                assoc_scores_rcm[i, in_valid_paths] = -10000
        
        return assoc_scores_rcm, assoc_scores_mcr
    
    def make_scores_boxes(self, assoc_scores_mcr, assoc_scores_rcm, frame_count, tracks_rider_dets, tracks_motor_dets, rider_all_paths, motor_all_paths, tracks_rider_tids, tracks_motor_tids):
        # The function sets the assoc_scores_mcr and the assoc_scores_rcm variables
        # self.boxes_motors is of shape (T, M, 4) and tracks_motor_dets is of shape (N, T)
        curr_motor_boxes = []
        start = 0
        for i in range(len(motor_all_paths)):
            to_append = self.boxes_motors[np.arange(start, start + frame_count, dtype=np.int32), tracks_motor_dets[i]]
            to_append[tracks_motor_dets[i] == -1] = np.zeros(4).astype(np.uint8)
            curr_motor_boxes.append(to_append)
        curr_motor_boxes = np.array(curr_motor_boxes)

        curr_rider_boxes = []
        for i in range(len(rider_all_paths)):
            to_append = self.boxes_riders[np.arange(start, start + frame_count, dtype=np.int32), tracks_rider_dets[i]]
            to_append[tracks_rider_dets[i] == -1] = np.zeros(4).astype(np.uint8)
            curr_rider_boxes.append(to_append)
        curr_rider_boxes = np.array(curr_rider_boxes)

        print(curr_motor_boxes.shape) # these are of shape (N, T, 4)
        print(curr_rider_boxes.shape) # these are of shape (M, T, 4)
        
        assoc_scores = np.zeros((len(rider_all_paths), len(motor_all_paths)))
        
        # assuming t = 1 for now only, will change later
        for t in range(frame_count):
            for i in range(len(rider_all_paths)):
                if(tracks_rider_dets[i, t] == -1):
                    continue
                iou_array = np.zeros(len(motor_all_paths))
                rider_box = curr_rider_boxes[i, t]
                motor_boxes = curr_motor_boxes[:, t]
                motor_dets = tracks_motor_dets[:, t]
                # find the iou between rider box and all motor_boxes vectorized
                for j in range(len(motor_all_paths)):
                    motor_box = motor_boxes[j]
                    if(motor_dets[j] == -1):
                        iou_array[j] = 0
                    else:
                        # print(rider_box)
                        # print(motor_box)
                        intersection = np.maximum(0, np.minimum(rider_box[2], motor_box[2]) - np.maximum(rider_box[0], motor_box[0])) * np.maximum(0, np.minimum(rider_box[3], motor_box[3]) - np.maximum(rider_box[1], motor_box[1]))
                        union = (rider_box[2] - rider_box[0]) * (rider_box[3] - rider_box[1]) + (motor_box[2] - motor_box[0]) * (motor_box[3] - motor_box[1]) - intersection
                        iou_array[j] = intersection / union
                        print(iou_array)
                        # print(iou_array[j], intersection, union)
                        # iou_array[np.isnan(iou_array)] = 0
                        # iou_array[np.isinf(iou_array)] = 0
                        iou_array[iou_array < 0] = -1000
                
                if(np.max(iou_array) > 0):
                    idx = np.argmax(iou_array)
                    motor_det_id = motor_dets[idx]
                    iou_array[motor_dets == motor_det_id] = 1
                    iou_array[motor_dets != motor_det_id] = -1
                assoc_scores[i, :] = iou_array
                print(assoc_scores)
            
        assoc_scores[assoc_scores <= 0] = -1000
        
        for i in range(len(curr_rider_boxes)):
            curr_rider_id = tracks_rider_tids[i]
            if(curr_rider_id in self.curr_assocs_inv):
                assoc_motor_id = self.curr_assocs_inv[curr_rider_id]
                in_valid_paths = tracks_motor_tids != assoc_motor_id
                assoc_scores[i, in_valid_paths] = -1000
        
        assoc_scores_mcr = assoc_scores
        assoc_scores_rcm = assoc_scores
        
        return assoc_scores_rcm, assoc_scores_mcr

    def make_problem(self, model, rider_vars, rider_all_paths, rider_costs, motor_vars, motor_all_paths, motor_costs, obj):
        # Quickly return in case there is no cross_association information
        if(len(rider_all_paths) == 0 or len(motor_all_paths) == 0):
            model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
            return []
        
        assoc_variables = []
        aux_variables = []
        # Add len(tracks_rider) * len(motor_vars) constraints to the problem
        for i in range(len(rider_all_paths)):
            rider_assocs = []
            rider_aux_var = []
            for j in range(len(motor_vars)):
                rider_assocs.append(model.addVar(vtype=gp.GRB.BINARY, name=f"assoc_{i}_{j}"))
                rider_aux_var.append(model.addVar(vtype=gp.GRB.BINARY, name=f"aux_{i}_{j}")) 
            assoc_variables.append(rider_assocs)
            aux_variables.append(rider_aux_var)

        # Add the constraint that every rider can be bound to only one motorcycle
        for i in range(len(rider_all_paths)):
            model.addConstr(gp.quicksum(assoc_variables[i]) <= 1)
        
        for i in range(len(rider_all_paths)):
            for j in range(len(motor_all_paths)):
                model.addConstr(aux_variables[i][j] == rider_vars[i] * motor_vars[j])
                model.addConstr(assoc_variables[i][j] * (1 - aux_variables[i][j]) == 0)
        
        # The problem
        # TODO: Vectorize Somehow
        # Make a cost array of shape (len(tracks_rider), len(motor_vars)) representing the association scores (b/w [0,1])
        assoc_scores_mcr = np.zeros((len(rider_all_paths), len(motor_vars))) + self.eps
        assoc_scores_rcm = np.zeros((len(rider_all_paths), len(motor_vars))) + self.eps
        def get_property(obj, attr):
            return getattr(obj, attr)

        get_property_vectorized = np.vectorize(get_property)
        
        # Consider the last n frames of all the hypothesis
        frame_count = self.frame_num
        start = 0
        if(frame_count > self.last_n_frames):
            frame_count = self.last_n_frames 
            start += self.frame_num - self.last_n_frames

        tracks_rider_dets = np.zeros((len(rider_all_paths), frame_count), dtype=np.int32)
        tracks_motor_dets = np.zeros((len(motor_all_paths), frame_count), dtype=np.int32)
        
        tracks_rider_tids = np.zeros((len(rider_all_paths)), dtype=np.int32)
        tracks_motor_tids = np.zeros((len(motor_all_paths)), dtype=np.int32)

        for i in range(len(rider_all_paths)):
            rider = rider_all_paths[i]
            rider_det_ids = np.array(get_property_vectorized(rider, "det_id"))
            rider_frame_num = np.array(get_property_vectorized(rider, "frame_num")) - 1
            rider_dets_ = np.ones(frame_count, dtype=np.int32) * -1
            first_frame_rider = rider_frame_num[0]
            if(first_frame_rider < self.frame_num - self.last_n_frames):
                first_frame_rider = - self.last_n_frames
                rider_dets_ = rider_det_ids[first_frame_rider:]
            else:
                rider_dets_[-self.frame_num + first_frame_rider:] = rider_det_ids
            tracks_rider_dets[i] = rider_dets_
            tracks_rider_tids[i] = rider[-1].track_id
        
        for i in range(len(motor_all_paths)):
            motor = motor_all_paths[i]
            motor_det_ids = np.array(get_property_vectorized(motor, "det_id"))
            motor_frame_num = np.array(get_property_vectorized(motor, "frame_num")) - 1
            motor_dets_ = np.ones(frame_count, dtype=np.int32) * -1
            first_frame_motor = motor_frame_num[0]
            if(first_frame_motor < self.frame_num - self.last_n_frames):
                first_frame_motor = - self.last_n_frames
                motor_dets_ = motor_det_ids[first_frame_motor:]
            else:
                motor_dets_[-self.frame_num + first_frame_motor: ] = motor_det_ids
            tracks_motor_dets[i] = motor_dets_
            tracks_motor_tids[i] = motor[-1].track_id

        # Fill the assoc_scores_mcr and assoc_scores_rcm
        assoc_scores_rcm, assoc_scores_mcr = self.make_scores(assoc_scores_mcr, assoc_scores_rcm, frame_count, tracks_rider_dets, tracks_motor_dets, rider_all_paths, motor_all_paths, tracks_rider_tids, tracks_motor_tids)
        # assoc_scores_rcm, assoc_scores_mcr = self.make_scores_boxes(assoc_scores_mcr, assoc_scores_rcm, frame_count, tracks_rider_dets, tracks_motor_dets, rider_all_paths, motor_all_paths, tracks_rider_tids, tracks_motor_tids)
        print(assoc_scores_mcr)
        print(assoc_scores_rcm)

        time_up = time.time()

        for i in range(len(rider_all_paths)):
            for j in range(len(motor_all_paths)):
                rider_track = rider_all_paths[i]
                motor_hypothesis = motor_all_paths[j]
                time_ = time.time()
                print(self.assoc_weight)
                obj += assoc_scores_mcr[i, j] * assoc_variables[i][j] * self.assoc_weight
                obj += assoc_scores_rcm[i, j] * assoc_variables[i][j] * self.assoc_weight

                print(rider_all_paths[i])
                print(motor_all_paths[j])
                print(rider_costs[i])
                print(motor_costs[j])
                print(assoc_scores_mcr[i][j])
                print(assoc_scores_rcm[i][j])
                print("Sum : ", self.assoc_weight * (assoc_scores_rcm[i][j] + assoc_scores_mcr[i][j]) + self.motor_obj_weight*(motor_costs[j]) + self.rider_obj_weight*(rider_costs[i]))
                print()

        print(time.time() - time_up)
        model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
        return assoc_variables

    def n_scan_pruning(self, model, rider_vars, motor_vars, rider_all_paths, motor_all_paths, rider_obj, motor_obj, rider_costs, motor_costs):
        # Solve joint tracking problem and association
        # model = gp.Model()
        # rider_vars, rider_all_paths, rider_obj, rider_costs = self.rider_tracker.solve_mwis_problem(model)
        # motor_vars, motor_all_paths, motor_obj, motor_costs = self.motor_tracker.solve_mwis_problem(model)
        tracks_rider, tracks_motor = [], []
        # if(not len(rider_all_paths) or not len(motor_all_paths)):
        #     return
        time0 = time.time()
        obj = self.rider_obj_weight * rider_obj + self.motor_obj_weight * motor_obj
        assoc_variables = self.make_problem(model, rider_vars, rider_all_paths, rider_costs, motor_vars, motor_all_paths, motor_costs, obj)
        self.make_problem_time.append(time.time() - time0)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(motor_vars)):
                if(motor_vars[i].x):
                    tracks_motor.append(motor_all_paths[i])

        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(rider_vars)):
                if(rider_vars[i].x):
                    tracks_rider.append(rider_all_paths[i])

        print("Tracks Rider")
        for tr in tracks_rider:
            print(tr)
        print("Tracks Motor")
        for tm in tracks_motor:
            print(tm)
        print()
        associations_riders = np.ones(len(tracks_rider)).astype(np.int32) * -1
        for i in range(len(assoc_variables)):
            for j in range(len(assoc_variables[i])):
                if(assoc_variables[i][j].x == 1):
                    for k in range(len(tracks_rider)):
                        if(tracks_rider[k] == rider_all_paths[i]):
                            rider_idx = k
                    for k in range(len(tracks_motor)):
                        if(tracks_motor[k] == motor_all_paths[j]):
                            motor_idx = k
                    associations_riders[rider_idx] = motor_idx
                    # print("Rider : ", rider_all_paths[i])
                    # print("Motor : ", motor_all_paths[j])
                    # print()

        self.rider_tracker.after_n_scan(tracks_rider, tracked=True)
        self.motor_tracker.after_n_scan(tracks_motor, tracked=True)

        self.update_curr_tracks(tracks_rider, tracks_motor, associations_riders)
        self.print_tracks(self.curr_rider_tracks)
        self.print_tracks(self.curr_motor_tracks)
        print()
        print(self.curr_assocs)
        # if(self.frame_num == 3):
            # exit(0)

    def update_curr_tracks(self, tracks_rider, tracks_motor, associations_riders):
        for i, r_track in enumerate(tracks_rider):
            tid = r_track[-1].track_id
            try:
                self.curr_rider_tracks[tid].extend([copy.deepcopy(r_track[-1])])
            except:
                self.curr_rider_tracks[tid] = r_track
        
        for i, m_track in enumerate(tracks_motor):
            tid = m_track[-1].track_id
            try:
                self.curr_motor_tracks[tid].extend([copy.deepcopy(m_track[-1])])
            except:
                self.curr_motor_tracks[tid] = m_track

        for i in range(len(associations_riders)):
            if(associations_riders[i] == -1):
                continue
            rider_tid = tracks_rider[i][-1].track_id
            motor_tid = tracks_motor[associations_riders[i]][-1].track_id
            try:
                self.curr_assocs[motor_tid][rider_tid] = 1
            except:
                self.curr_assocs[motor_tid] = {rider_tid : 1}

            try:
                assoc_motor_tid = self.curr_assocs_inv[rider_tid]
                assert assoc_motor_tid == motor_tid, f"Association ID of rider with {rider_tid} changed from {assoc_motor_tid} to {motor_tid}"
            except:
                self.curr_assocs_inv[rider_tid] = motor_tid

    
    def update_curr_tracks_old(self, tracks_rider, tracks_motor, associations_riders):
        rider_indices_present = np.ones(len(tracks_rider)).astype(np.int32) * -1
        # print("RIDER UPDATE")
        for i, r_track in enumerate(tracks_rider):
            for j, rider in enumerate(self.curr_rider_tracks):
                if(r_track[0].frame_num == rider[0].frame_num and r_track[0].det_id == rider[0].det_id and r_track[0].det_id != -1):
                    self.curr_rider_tracks[j] = copy.deepcopy(r_track)
                    rider_indices_present[i] = j
                elif(len(r_track) >= self.max_tree_depth and len(rider) >= self.max_tree_depth):
                    # Check the last self.max_tree_depth nodes for both
                    same = False
                    for k in range(1, self.max_tree_depth):
                        if(rider[-k].det_id != -1 and rider[-k].det_id == r_track[-k-1].det_id and rider[-k].frame_num == r_track[-k-1].frame_num):
                            same = True
                            break
                    if(same):
                        # print(self.curr_rider_tracks[j])
                        # print(r_track)
                        self.curr_rider_tracks[j].extend([copy.deepcopy(r_track[-1])])
                        # print("UPDATED : ", self.curr_rider_tracks[j])
                        rider_indices_present[i] = j
        
        addition_track_ids = len(self.curr_rider_tracks)
        for i, r_track in enumerate(tracks_rider):
            if(rider_indices_present[i] == -1):
                self.curr_rider_tracks.extend([copy.deepcopy(r_track)])
                rider_indices_present[i] = addition_track_ids
                addition_track_ids += 1
        
        # print("MOTOR UPDATE")
        motor_indices_present = np.ones(len(tracks_motor)).astype(np.int32) * -1
        for i, m_track in enumerate(tracks_motor):
            for j, motor in enumerate(self.curr_motor_tracks):
                if(m_track[0].frame_num == motor[0].frame_num and m_track[0].det_id == motor[0].det_id and m_track[0].det_id != -1):
                    self.curr_motor_tracks[j] = copy.deepcopy(m_track)
                    motor_indices_present[i] = j
                elif(len(m_track) >= self.max_tree_depth and len(motor) >= self.max_tree_depth):
                    # Check the last self.max_tree_depth nodes for both
                    same = False
                    for k in range(1, self.max_tree_depth):
                        if(motor[-k].det_id != -1 and motor[-k].det_id == m_track[-k-1].det_id and motor[-k].frame_num == m_track[-k-1].frame_num):
                            same = True
                            break
                    if(same):
                        # print(self.curr_motor_tracks[j])
                        # print(m_track)
                        self.curr_motor_tracks[j].extend([copy.deepcopy(m_track[-1])])
                        # print("UPDATED : ", self.curr_motor_tracks[j])
                        motor_indices_present[i] = j
        
        addition_track_ids = len(self.curr_motor_tracks)
        for i, m_track in enumerate(tracks_motor):
            if(motor_indices_present[i] == -1):
                self.curr_motor_tracks.extend([copy.deepcopy(m_track)])
                motor_indices_present[i] = addition_track_ids
                addition_track_ids += 1

        # print(motor_indices_present)
        # print(associations_riders)
        for i in range(len(associations_riders)):
            if(associations_riders[i] == -1):
                continue
            rider_idx_in_curr_tracks = rider_indices_present[i]
            motor_idx_in_curr_tracks = motor_indices_present[associations_riders[i]]
            try:
                self.curr_assocs[motor_idx_in_curr_tracks][rider_idx_in_curr_tracks] = 1
            except:
                self.curr_assocs[motor_idx_in_curr_tracks] = {rider_idx_in_curr_tracks : 1}
        

    def give_tracks(self):
        model = gp.Model()
        rider_vars, rider_all_paths, rider_obj, rider_costs = self.rider_tracker.solve_mwis_problem(model)
        motor_vars, motor_all_paths, motor_obj, motor_costs = self.motor_tracker.solve_mwis_problem(model)
        tracks_rider, tracks_motor, assocs = [], [], []
        
        print("Final Motor paths : ")
        print(motor_all_paths)
        
        obj = self.rider_obj_weight * rider_obj + self.motor_obj_weight * motor_obj
        assoc_variables = self.make_problem(model, rider_vars, rider_all_paths, rider_costs, motor_vars, motor_all_paths, motor_costs, obj)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(motor_vars)):
                if(motor_vars[i].x):
                    tracks_motor.append(motor_all_paths[i])
        
        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(rider_vars)):
                if(rider_vars[i].x):
                    tracks_rider.append(rider_all_paths[i])

        print("Tracks Rider")
        for tr in tracks_rider:
            print(tr)
        print("Tracks Motor")
        for tm in tracks_motor:
            print(tm)

        associations_riders = np.ones(len(tracks_rider)).astype(np.int32) * -1
        for i in range(len(assoc_variables)):
            for j in range(len(assoc_variables[i])):
                if(assoc_variables[i][j].x == 1):
                    for k in range(len(tracks_rider)):
                        if(tracks_rider[k] == rider_all_paths[i]):
                            rider_idx = k
                    for k in range(len(tracks_motor)):
                        if(tracks_motor[k] == motor_all_paths[j]):
                            motor_idx = k
                    associations_riders[rider_idx] = motor_idx
        
        # IMP
        # For n_scan=1, we dont need this
        # self.update_curr_tracks(tracks_rider, tracks_motor, associations_riders)

        assoc_riders = np.ones(len(self.curr_rider_tracks), dtype=np.int32) * -1
        assoc_motors = np.arange(len(self.curr_motor_tracks), dtype=np.int32)

        for m_id, rider_dict in self.curr_assocs.items():
            for r_id, _ in rider_dict.items():
                assoc_riders[r_id] = m_id

        # curr_rider_ = 0
        # for i in range(len(assoc_variables)):
        #     for j in range(len(assoc_variables[i])):
        #         if(assoc_variables[i][j].x == 1):
        #             aid = -1
        #             for k in range(len(tracks_motor)):
        #                 if(motor_all_paths[j] == tracks_motor[k]):
        #                     aid = k
        #                     break
        #             rider_idx = -1
        #             for k in range(len(tracks_rider)):
        #                 if(rider_all_paths[i] == tracks_rider[k]):
        #                     rider_idx = k
        #                     break
        #             assoc_riders[rider_idx] = aid
        #             curr_rider_ += 1
        #             print("Rider : ", rider_all_paths[i])
        #             print("Motor : ", motor_all_paths[j])
        #             print()
        self.print_tracks(self.curr_rider_tracks)
        self.print_tracks(self.curr_motor_tracks)
        print(assoc_riders)
        print(assoc_motors)
        print(self.curr_assocs)
        out_rider = self.rider_tracker.give_tracks(self.curr_rider_tracks, assoc_riders)
        out_motor = self.motor_tracker.give_tracks(self.curr_motor_tracks, assoc_motors)

        motor_rider_count = np.zeros(len(self.curr_motor_tracks), dtype=np.int32)
        for i in range(len(motor_rider_count)):
            motor_rider_count[i] = len(np.where(assoc_riders == i)[0])

        print(motor_rider_count)
        return out_rider, out_motor, motor_rider_count
    
    def print_tracks(self, track):
        for tr in track:
            print(tr)
    
    def null(self, *args, **kwargs):
        pass
import numpy as np
from utils.kalman_filter import KalmanFilterXYWH
from fast_reid.fast_reid_interfece import FastReIDInterface
import torch
import copy
import gurobipy as gp
from gurobipy import GRB
from scipy.interpolate import interp1d
from pulp import *
import time
from tracker_reid.offline_tracker.tracker_dists import TrackerDists
from .node import Node
from .detection import Detection

REID_CONFIG = '/Users/keshavgupta/desktop/CVIT/TrafficViolations/tracker_reid/fast_reid/configs/Market1501/sbs_R101-ibn.yml'
# REID_CONFIG = '/home2/keshav06/TrafficViolations/tracker_reid/fast_reid/configs/Market1501/sbs_R101-ibn.yml'
REID_WEIGHTS = '/Users/keshavgupta/desktop/CVIT/TrafficViolations/weights/market_sbs_R101-ibn.pth'
# REID_WEIGHTS = '/home2/keshav06/TrafficViolations/weights/market_sbs_R101-ibn.pth'
REID_MAPPING_FILE = "/Users/keshavgupta/desktop/CVIT/TrafficViolations/simil_off_diag.npy"

# TODO : Can make this MUCH MORE memory efficent as Node contains detections and we are making many copies of a given node.
# We can avoid that by having a global memory for all the node.detections

class AssocTracker():
    def __init__(self):
        self.logger = print #self.null
        self.frame_num = 0
        self.prune_after = 2
        self.rider_tracker = Tracker("rider", self.prune_after, self.logger)
        self.motor_tracker = Tracker("motor", self.prune_after, self.logger)
        self.masks_riders = None
        self.cross_masks_riders = None
        self.masks_motors = None
        self.cross_masks_motors = None
        self.mask_shape = None
    
    def append_to_array(self, arr, elem):
        num_dets = elem.shape[0]
        max_dets_till_now = arr.shape[1] if arr is not None else 0
        new_max_dets = max(num_dets, max_dets_till_now)
        if(arr is not None):
            new = np.zeros((arr.shape[0] + 1, new_max_dets, *elem.shape[1:])).astype(np.uint8)
            new[:-1, :max_dets_till_now] = arr
            new[-1, :elem.shape[0]] = elem
        else:
            new = elem[None]
        return new
    
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def track(self, detections, masks, cross_masks, frame):
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
        self.frame_num += 1
        
        cls = detections[:, -1]
        # Need for defining the association costs while making the problem
        self.masks_riders = self.append_to_array(self.masks_riders, masks[cls == 0].astype(np.uint8))
        self.masks_motors = self.append_to_array(self.masks_motors, masks[cls == 1].astype(np.uint8))
        self.cross_masks_riders = self.append_to_array(self.cross_masks_riders, cross_masks[cls == 0].astype(np.uint8))
        self.cross_masks_motors = self.append_to_array(self.cross_masks_motors, cross_masks[cls == 1].astype(np.uint8))

        print(self.masks_riders.shape)
        # exit(0)
        # self.masks_riders.append(masks[cls == 0])
        # self.masks_motors.append(masks[cls == 1])
        # self.cross_masks_riders.append(cross_masks[cls == 0])
        # self.cross_masks_motors.append(cross_masks[cls == 1])

        if(self.mask_shape is None):
            self.mask_shape = masks[0].shape

        rider_dets = detections[cls == 0][:, :-1]
        motor_dets = detections[cls == 1][:, :-1]
        
        if(self.frame_num % self.prune_after == 0):
            # make gp model
            model_rider = gp.Model()
            model_motor = gp.Model()
        else:
            model_rider = None
            model_motor = None
        
        rider_feats, rider_vars, rider_all_paths, rider_obj, rider_costs = self.rider_tracker.track(rider_dets, frame, model_rider)
        motor_feats, motor_vars, motor_all_paths, motor_obj, motor_costs = self.motor_tracker.track(motor_dets, frame, model_motor)
        if(self.frame_num % self.prune_after == 0):
            self.n_scan_pruning(model_rider, model_motor, rider_vars, motor_vars, rider_all_paths, motor_all_paths, rider_obj, motor_obj, rider_costs, motor_costs)

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

    def make_problem(self, model_motor, tracks_rider, motor_vars, motor_all_paths, motor_obj, motor_costs):
        # Add len(tracks_rider) * len(motor_vars) constraints to the problem
        assoc_variables = []
        for i in range(len(tracks_rider)):
            rider_assocs = []
            for j in range(len(motor_vars)):
                rider_assocs.append(model_motor.addVar(vtype=gp.GRB.BINARY, name=f"assoc_{i}_{j}"))
            assoc_variables.append(rider_assocs)

        # Add the constraint that every rider can be bound to only one motorcycle
        for i in range(len(tracks_rider)):
            model_motor.addConstr(gp.quicksum(assoc_variables[i]) <= 1)
        
        # The problem
        # TODO: Vectorize Somehow
        # Make a cost array of shape (len(tracks_rider), len(motor_vars)) representing the association scores (b/w [0,1])
        obj = 0 # motor_obj
        assoc_scores = np.zeros((len(tracks_rider), len(motor_vars)))
        def get_property(obj, attr):
            return getattr(obj, attr)

        get_property_vectorized = np.vectorize(get_property)

        tracks_rider_dets = np.zeros((len(tracks_rider), self.frame_num), dtype=np.int32)
        tracks_motor_dets = np.zeros((len(motor_all_paths), self.frame_num), dtype=np.int32)

        for i in range(len(tracks_rider)):
            rider = tracks_rider[i]
            rider_det_ids = np.array(get_property_vectorized(rider, "det_id"))
            rider_frame_num = np.array(get_property_vectorized(rider, "frame_num")) - 1
            rider_dets_ = np.ones(self.frame_num, dtype=np.int32) * -1
            rider_dets_[rider_frame_num[0]:] = rider_det_ids
            tracks_rider_dets[i] = rider_dets_
        
        for i in range(len(motor_all_paths)):
            motor = motor_all_paths[i]
            motor_det_ids = np.array(get_property_vectorized(motor, "det_id"))
            motor_frame_num = np.array(get_property_vectorized(motor, "frame_num")) - 1
            motor_dets_ = np.ones(self.frame_num, dtype=np.int32) * -1
            motor_dets_[motor_frame_num[0]:] = motor_det_ids
            tracks_motor_dets[i] = motor_dets_

        # print("dets formed", tracks_motor_dets.shape)
        # print(tracks_motor_dets)
        # index = tracks_motor_dets[]
        # self.masks_motors is of shape (T, M, H, W) and tracks_motor_dets is of shape (N, T)
        curr_motor_masks = []
        for i in range(len(motor_all_paths)):
            to_append = self.masks_motors[np.arange(self.frame_num, dtype=np.int32), tracks_motor_dets[i]]
            to_append[tracks_motor_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_motor_masks.append(to_append)
        curr_motor_masks = np.array(curr_motor_masks)

        curr_cross_rider_masks = []
        for i in range(len(tracks_rider)):
            to_append = self.cross_masks_riders[np.arange(self.frame_num, dtype=np.int32), tracks_rider_dets[i]]
            to_append[tracks_rider_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_cross_rider_masks.append(to_append)
        curr_cross_rider_masks = np.array(curr_cross_rider_masks)

        # tracks_rider_dets_ = tracks_rider_dets[:, None]
        # tracks_motor_dets_ = tracks_motor_dets[None]
        # tracks_rider_dets_ = tracks_rider_dets_ * np.ones_like(tracks_motor_dets_)
        # tracks_motor_dets_ = tracks_motor_dets_ * np.ones_like(tracks_rider_dets_)

        # for i in range(len(curr_cross_rider_masks)):
        #     scores = np.zeros_like(tracks_motor_dets)
        #     intersection = (curr_cross_rider_masks[i][None] & curr_motor_masks).reshape(len(motor_all_paths), self.frame_num, -1).sum(-1)
        #     union = (curr_cross_rider_masks[i][None] | curr_motor_masks).reshape(len(motor_all_paths), self.frame_num, -1).sum(-1)
        #     cond = (tracks_motor_dets != -1) * (tracks_rider_dets[i][None] != -1)
        #     scores[cond] = intersection[cond] / union[cond] - 0.5
        #     assoc_scores[i] = self.sigmoid(np.sum(scores, -1))

        for i in range(len(curr_motor_masks)):
            scores = np.zeros_like(tracks_rider_dets).astype(np.float32)
            intersection = (curr_cross_rider_masks & curr_motor_masks[i][None]).reshape(len(tracks_rider), self.frame_num, -1).sum(-1)
            union = (curr_cross_rider_masks | curr_motor_masks[i][None]).reshape(len(tracks_rider), self.frame_num, -1).sum(-1)
            cond = (tracks_motor_dets[i][None] != -1) * (tracks_rider_dets != -1)
            scores[cond] = np.float32(intersection[cond]) / np.float32(union[cond]) - 0.5
            print(motor_all_paths[i], scores)
            scores[np.isnan(scores)] = 0
            scores[np.isinf(scores)] = 0
            if(np.isnan(scores).sum() or np.isinf(scores).sum()):
                exit(0)
            # assoc_scores[:, i] = np.sum(self.sigmoid(np.cumsum(scores[:,  :], -1)), -1)
            assoc_scores[:, i] = np.sum(scores[:,  :], -1)
        # for i in range(len(motor_all_paths)):
        #     scores = np.zeros(self.frame_num)
        #     for j in range(len(tracks_rider)):
        #         print(f"{i}-{j}")
        #         to_append_motor = self.masks_motors[np.arange(self.frame_num, dtype=np.int32), tracks_motor_dets[i]]
        #         to_append_motor[tracks_motor_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
        #         to_append_rider = self.cross_masks_riders[np.arange(self.frame_num, dtype=np.int32), tracks_rider_dets[j]]
        #         to_append_rider[tracks_rider_dets[j] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
        #         print(to_append_motor.shape)
        #         print(to_append_rider.shape)
        #         intersection = (to_append_rider & to_append_motor).reshape(self.frame_num, -1).sum(-1)
        #         union = (to_append_rider | to_append_motor).reshape(self.frame_num, -1).sum(-1)
        #         cond = (tracks_motor_dets[i] != -1) * (tracks_rider_dets[j] != -1)
        #         scores[cond] = intersection[cond] / union[cond] - 0.5
        #         assoc_scores[j, i] = self.sigmoid(np.sum(scores, -1))

        time_up = time.time()
        for i in range(len(tracks_rider)):
            for j in range(len(motor_all_paths)):
                rider_track = tracks_rider[i]
                motor_hypothesis = motor_all_paths[j]
                time_ = time.time()
                # assoc_scores[i, j] = self.calc_assoc_score(rider_track, motor_hypothesis)
                if(assoc_scores[i, j] > 0.5):
                    print("Rider Track : ", rider_track)
                    print("Motor Track : ", motor_hypothesis, motor_costs[j])
                    print(assoc_scores[i, j])
                obj += assoc_scores[i, j] * assoc_variables[i][j] * motor_vars[j]
        print(time.time() - time_up)
        model_motor.setObjective(obj, sense=gp.GRB.MAXIMIZE)
        return assoc_variables

    def n_scan_pruning(self, model_rider, model_motor, rider_vars, motor_vars, rider_all_paths, motor_all_paths, rider_obj, motor_obj, rider_costs, motor_costs):
        # Solve the rider tracking problem first
        tracks_rider, tracks_motor = [], []

        model_rider.setObjective(rider_obj, sense=gp.GRB.MAXIMIZE)
        model_rider.optimize()

        if model_rider.status == gp.GRB.OPTIMAL:
            for i in range(len(rider_vars)):
                if(rider_vars[i].x):
                    tracks_rider.append(rider_all_paths[i])

        assoc_variables = self.make_problem(model_motor, tracks_rider, motor_vars, motor_all_paths, motor_obj, motor_costs)
        model_motor.optimize()

        if model_motor.status == gp.GRB.OPTIMAL:
            for i in range(len(motor_vars)):
                if(motor_vars[i].x):
                    tracks_motor.append(motor_all_paths[i])
        
        print("Tracks Rider")
        for tr in tracks_rider:
            print(tr)
        print("Tracks Motor")
        for tm in tracks_motor:
            print(tm)

        for i in range(len(assoc_variables)):
            for j in range(len(assoc_variables[i])):
                if(assoc_variables[i][j].x == 1):
                    print(tracks_rider[i], motor_all_paths[j])
        # exit(0)
        self.rider_tracker.after_n_scan(tracks_rider)
        self.motor_tracker.after_n_scan(tracks_motor)
        
    def give_tracks(self):
        model_rider = gp.Model()
        model_motor = gp.Model()
        rider_vars, rider_all_paths, rider_obj, rider_costs = self.rider_tracker.solve_mwis_problem(model_rider)
        motor_vars, motor_all_paths, motor_obj, motor_costs = self.motor_tracker.solve_mwis_problem(model_motor)
        tracks_rider, tracks_motor = [], []

        model_rider.setObjective(rider_obj, sense=gp.GRB.MAXIMIZE)
        model_rider.optimize()

        if model_rider.status == gp.GRB.OPTIMAL:
            for i in range(len(rider_vars)):
                if(rider_vars[i].x):
                    tracks_rider.append(rider_all_paths[i])

        assoc_variables = self.make_problem(model_motor, tracks_rider, motor_vars, motor_all_paths, motor_obj, motor_costs)
        model_motor.optimize()

        if model_motor.status == gp.GRB.OPTIMAL:
            for i in range(len(motor_vars)):
                if(motor_vars[i].x):
                    tracks_motor.append(motor_all_paths[i])
        
        print("Tracks Rider")
        for tr in tracks_rider:
            print(tr)
        print("Tracks Motor")
        for tm in tracks_motor:
            print(tm)

        out_rider = self.rider_tracker.give_tracks(tracks_rider)
        out_motor = self.motor_tracker.give_tracks(tracks_motor)

        return out_rider, out_motor
    
    def null(self, *args, **kwargs):
        pass
        

class Tracker():
    def __init__(self, name, prune_after, logger):
        self.name = name
        self.detections = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kalman_filter = KalmanFilterXYWH()
        self.motion_gating_threshold = 20
        self.prune_after = prune_after
        self.max_det_per_frame = 50
        self.appearance_thresh = 0.975
        self.miss_prob = np.log(1 - 0.9)
        self.logger = logger
        self.frame_num = 0
        self.reid_feat_dim = None
        self.trees = []
        self.reid_encoder = FastReIDInterface(config_file=REID_CONFIG, weights_path=REID_WEIGHTS, device=self.device)
        
        # self.reid_score_mapping_file = REID_MAPPING_FILE
        # simil_off_diag = np.load(self.reid_score_mapping_file)
        # freq, bins = np.histogram(simil_off_diag, bins=np.linspace(0.9, 1.0, 50))
        # freq = freq / np.sum(freq)
        # cdf = np.cumsum(freq)

        bins = np.linspace(0.90, 1.0, num=11)
        if(name == "rider"):
            cdf = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.2, 0.7, 0.8, 0.9, 1.0])
        elif(name == "motor"):
            # Dont use the ReID model
            # TODO : Remove the ReID Encoder Inference as well
            cdf = np.ones(11)
        self.min_reid_score = bins[0]
        self.reid_normalize_fn = interp1d(bins, cdf, kind='linear')
        self.tracker_dists = TrackerDists(self.reid_normalize_fn, self.min_reid_score, self.motion_gating_threshold, self.miss_prob, self.logger)

    def track(self, detections, frame, problem):
        '''
        detections are the new detections in the frame. Assuming a new scan from the sensor. It is a numpy array of shape (N, 5)
        representing the coordinates of the bounding box detection in (xyxy format) as well as the confidence of the detection.
        
        frame is a numpy array of shape (H, W, 3) representing the image of the environment

        problem is an instance of gp.Model, is None if the frame is not a multiple of prune_after

        The function updates the Tracker with the new detections
        '''
        # We need to store the detections as well as the Re-ID features.
        # Lets compute the reid features for all the detections
        self.frame_num += 1
        reid_features = self.reid_encoder.inference(image=frame, detections=detections)
        # reid_features = np.random.randn(len(detections), 10)
        if(len(reid_features) == 0):
            return []

        self.reid_feat_dim = reid_features.shape[1]
        all_dets = []
        all_nodes = []
        for i, det in enumerate(detections):
            detection = Detection(det, reid_features[i]/np.linalg.norm(reid_features[i]))
            node = Node(self.frame_num, detection, i)
            all_nodes.append(node)
            all_dets.append(detection)
        
        # ### Return for viz
        # det_feats = []
        # for i in range(len(all_dets)):
        #     det_feats.append(all_dets[i].feature)

        # return det_feats
        
        # Make a null node for no detection
        node = Node(self.frame_num, Detection(np.zeros(5), np.zeros(self.reid_feat_dim)), -1)
        all_nodes.append(node)

        # Add the new nodes as the leaves of the existing trees
        count = 0
        for tree in self.trees:
            tree.add_children_to_all_leaves(all_nodes)
            count += 1

        # Prune the newly formed tracks
        self.prune_tracks()

        if(self.frame_num % self.prune_after == 0):
            variables, all_paths, obj, all_costs = self.n_scan_pruning(problem)
        else:
            variables, all_paths, obj, all_costs = None, None, None, None

        # self.logger("###############################")
        # all_paths = self.give_all_paths()
        # self.logger(f"{self.frame_num} : ", len(all_paths))
        # self.logger("###############################")

        # Make new trees corresponding to the nodes
        for node in all_nodes:
            if(node.det_id != -1):
                self.trees.append(copy.deepcopy(node))

        self.detections.append(all_dets)

        ### Return for viz
        det_feats = []
        for i in range(len(all_dets)):
            det_feats.append(all_dets[i].feature)

        return det_feats, variables, all_paths, obj, all_costs

    def give_tracks(self, tracks):
        '''
        returns the final tracks obtained after solving the mwis problem
        '''
        self.logger(tracks)
        out = {}
        # self.logger(tracks)
        for i in range(self.frame_num):
            out[i + 1] = []

        for i, track in enumerate(tracks):
            for node in track:
                out[node.frame_num].append((i, node.detection.bounding_box_xyxyc))
        
        return out

    def give_all_paths(self):
        '''
        gives all the paths in the all the trees
        '''

        # Append all the paths of all the trees in a list
        paths = []
        for tree in self.trees:
            paths = paths + tree.give_all_paths()
        
        for i, path in enumerate(paths):
            paths[i] = path[::-1]

        self.logger("Number of paths : ", len(paths))
        return paths
    
    # BOTTLENECK
    def build_graph_slow(self):
        '''
        builds the graph given all the trees formed till now. For every path in every tree, make a node in the graph and if there is any
        detection common to any paths then there is an edge between the 2 nodes in the graph.
        '''

        all_paths = self.give_all_paths()
        adj_mat = np.zeros((len(all_paths), len(all_paths)))

        def same_or_not(path1, path2):
            for node1 in path1:
                for node2 in path2:
                    if(node1.frame_num == node2.frame_num and node1.det_id == node2.det_id and node1.det_id != -1 and node2.det_id != -1):
                        return 1
            return 0

        # TODO : Vectorize this somehow
        for i in range(len(all_paths)):
            for j in range(len(all_paths)):
                # Fill with 1 if there is any node common in them
                path1 = all_paths[i]
                path2 = all_paths[j]
                if(same_or_not(path1, path2)):
                    adj_mat[i][j] = 1

        return all_paths, adj_mat
    
    def build_graph(self):
        '''
        builds the graph given all the trees formed till now. For every path in every tree, make a node in the graph and if there is any
        detection common to any paths then there is an edge between the 2 nodes in the graph.
        '''

        all_paths = self.give_all_paths()
        adj_mat = np.zeros((len(all_paths), len(all_paths)))
        if(len(all_paths) == 0):
            print("sdf")
            return all_paths, adj_mat
        
        max_path_len = max([len(path) for path in all_paths])
        path_array = np.zeros((len(all_paths), max_path_len), dtype=np.int32)
        
        for i, path in enumerate(all_paths):
            for j, node in enumerate(path):
                frame_num = node.frame_num
                det_id = node.det_id
                if(det_id != -1):
                    hash_val = frame_num * self.max_det_per_frame + det_id
                else:
                    hash_val = -1
                path_array[i, j] = hash_val
            
            for j in range(len(path), max_path_len):
                path_array[i, j] = -1
        
        # Check for intersection in the arrays with itself
        for i, path in enumerate(path_array):
            hash_vals = path[path != -1].tolist()
            intersection = np.any(np.isin(path_array, hash_vals), axis=1)
            adj_mat[i, :] = intersection

        return all_paths, adj_mat
    
    def get_costs_for_all_path_list(self, path_list):
        '''
        Returns the cost array of all the paths in the path_list
        '''
        cost = []
        for path in path_list:
            self.logger("Path : ", path)
            cost.append(self.tracker_dists.compute_dist_for_track(path))
            self.logger()
        
        return cost
    
    def n_scan_pruning(self, problem):
        '''
        Do N-Scan pruning on all the trees. Solve the MWIS problem first and then any hypothesis starting from
        frame K - N (where K is the current frame number) which is diverging from the true solution will be pruned.
        '''
        self.logger("||||||||||||||||N SCAN PRUNING||||||||||||||||||")
        return self.solve_mwis_problem(problem)
    
    def after_n_scan(self, tracks):
        
        # From every tree, there can be only one possible solution, hence a track in tracks list wont occur in 2 trees at once
        tree_track_idx = []
        self.logger("AFTER")
        self.logger(self.trees)
        for track in tracks:
            for i, tree in enumerate(self.trees):
            # Check if the track belongs to this tree or not
                if(tree.det_id == track[0].det_id and tree.frame_num == track[0].frame_num):
                    # Track blongs to the tree, prune all the other paths belonging to the tree
                    tree.prune_except(track)
                    self.logger()
                    tree_track_idx.append(i)
                    break
        self.logger("PRUNING DONE")
        self.logger("TREES NOW")
        self.logger(self.trees)
        # paths = self.give_all_paths()
        # for path in paths:
        #     self.logger(path)
        
        # self.logger(tree_track_idx)
        # Remove all the trees that dont belong to any track
        rest_trees = []
        i = 0
        while(len(self.trees)):
            tree = self.trees.pop(0)
            if(i in tree_track_idx):
                rest_trees.append(tree)
            else:
                tree.prune_all_children()
                # del tree
            i += 1
        
        self.trees = rest_trees

    def solve_mwis_problem_pulp(self):
        '''
        build the mwis problem and solve it
        returns the tracks formed
        '''
        
        all_paths, adj_mat = self.build_graph()
        all_costs = self.get_costs_for_all_path_list(all_paths)
        self.logger("**********MWIS**********")
        self.logger("ALL PATHS")
        for path in all_paths:
            self.logger(path)
        self.logger("ALL PATHS LEN : ", len(all_paths))
        # Corresponding to each node in the graph there is a binary integer variable {0, 1},
        # The cost of the path is the cost of the variable,
        # The constraints are for every edge in the graph between xi and xj => xi + xj <= 1 {i != j} (implying that we can't choose both together)

        prob = LpProblem("MWIS", LpMaximize)
        variables = [LpVariable(f"{i}", cat=const.LpBinary) for i, _ in enumerate(all_paths)]

        # The problem
        expr = 0
        for i in range(len(variables)):
            expr += variables[i] * all_costs[i]
        prob += expr

        count = 0
        # The constraints
        for i in range(len(variables)):
            adj_vector = adj_mat[i]
            neighbors = np.where(adj_vector == 1)[0]
            neighbors = neighbors[neighbors < i]
            for j in neighbors:
                # self.logger("MAKING THE PROBLEM : ", count)
                prob += variables[i] + variables[j] <= 1
                count += 1

        
        # Solve the problem
        tracks = []

        prob.solve(GUROBI_CMD(options=[("MIPgap", 0)]))
        # prob.solve()
        for i, var in enumerate(variables):
            if(var.varValue):
                tracks.append(all_paths[i])
        
        self.logger("Optimized Tracks")
        # self.logger(tracks)
        for track in tracks:
            self.logger(track)
        self.logger("**********MWIS**********")
        
        return tracks
    
    def solve_mwis_problem(self, problem):
        '''
        build the mwis problem and solve it
        returns the tracks formed
        '''
        
        all_paths, adj_mat = self.build_graph()
        all_costs = self.get_costs_for_all_path_list(all_paths)
        self.logger("**********MWIS**********")
        self.logger("ALL PATHS")
        for path in all_paths:
            self.logger(path)
        self.logger("ALL PATHS LEN : ", len(all_paths))
        # Corresponding to each node in the graph there is a binary integer variable {0, 1},
        # The cost of the path is the cost of the variable,
        # The constraints are for every edge in the graph between xi and xj => xi + xj <= 1 {i != j} (implying that we can't choose both together)

        variables = [problem.addVar(vtype=gp.GRB.BINARY, name=f"{self.name}_{i}") for i, _ in enumerate(all_paths)]

        # The problem
        obj = gp.quicksum(all_costs[i] * variables[i] for i in range(len(all_costs)))

        count = 0
        # The constraints
        for i in range(len(variables)):
            adj_vector = adj_mat[i]
            neighbors = np.where(adj_vector == 1)[0]
            neighbors = neighbors[neighbors < i]
            for j in neighbors:
                # self.logger("MAKING THE PROBLEM : ", count)
                problem.addConstr(variables[i] + variables[j] <= 1)
                count += 1
        
        return variables, all_paths, obj, all_costs
    
    def prune_tracks(self):
        '''
        prune the tracks that are not possible to avoid exponential explosion (based on kinematic and non_kinematic costs)
        '''
        # Get all the paths in all the trees and iterate over them one by one
        self.logger("$$$$$$$$$$$$$$$$$ PRUNE TRACKS $$$$$$$$$$$$$$$$$$$$")
        for tree in self.trees:
            tree.prune_paths([], self.tracker_dists.prune_condition_function)
    
    def null(self, *args, **kwargs):
        pass
import numpy as np
from utils.kalman_filter import KalmanFilterXYWH
from fast_reid.fast_reid_interfece import FastReIDInterface
import torch
import copy
from scipy.interpolate import interp1d
from pulp import *
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

class Tracker():
    def __init__(self):
        self.detections = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kalman_filter = KalmanFilterXYWH()
        self.motion_gating_threshold = 20
        self.prune_after = 3 # For now
        self.max_det_per_frame = 50
        self.appearance_thresh = 0.975
        self.miss_prob = np.log(1 - 0.9)
        self.logger = print #self.null
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
        cdf = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.2, 0.7, 0.8, 0.9, 1.0])
        self.min_reid_score = bins[0]
        self.reid_normalize_fn = interp1d(bins, cdf, kind='linear')
        self.tracker_dists = TrackerDists(self.reid_normalize_fn, self.min_reid_score, self.motion_gating_threshold, self.miss_prob, self.logger)

    def track(self, detections, frame):
        '''
        detections are the new detections in the frame. Assuming a new scan from the sensor. It is a numpy array of shape (N, 5)
        representing the coordinates of the bounding box detection in (xyxy format) as well as the confidence of the detection.
        
        frame is a numpy array of shape (H, W, 3) representing the image of the environment

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
            self.n_scan_pruning()

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

        return det_feats

    def give_tracks(self):
        '''
        returns the final tracks obtained after solving the mwis problem
        '''
        tracks = self.solve_mwis_problem()
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
    
    def n_scan_pruning(self):
        '''
        Do N-Scan pruning on all the trees. Solve the MWIS problem first and then any hypothesis starting from
        frame K - N (where K is the current frame number) which is diverging from the true solution will be pruned.
        '''
        self.logger("||||||||||||||||N SCAN PRUNING||||||||||||||||||")
        tracks = self.solve_mwis_problem()
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

    def solve_mwis_problem(self):
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
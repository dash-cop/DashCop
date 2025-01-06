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
from tracker.assoc_tracker.tracker_dists import TrackerDists
from .node import Node
from .detection import Detection
import lap # lapx>=0.5.2


REID_CONFIG = '/Users/keshavgupta/desktop/CVIT/TrafficViolations/tracker_reid/fast_reid/configs/Market1501/sbs_R101-ibn.yml'
# REID_CONFIG = '/home2/keshav06/TrafficViolations/tracker_reid/fast_reid/configs/Market1501/sbs_R101-ibn.yml'
REID_WEIGHTS = '/Users/keshavgupta/desktop/CVIT/TrafficViolations/weights/market_sbs_R101-ibn.pth'
# REID_WEIGHTS = '/home2/keshav06/TrafficViolations/weights/market_sbs_R101-ibn.pth'
REID_MAPPING_FILE = "/Users/keshavgupta/desktop/CVIT/TrafficViolations/simil_off_diag.npy"

class Tracker():
    def __init__(self, name, prune_after, logger, reid_encoder, max_tree_depth, cfg):
        self.name = name
        self.cfg = cfg
        self.detections = []
        self.kalman_filter = KalmanFilterXYWH()
        self.prune_after = prune_after
        self.max_det_per_frame = 50
        self.miss_prob = np.log(1 - 0.9)
        self.next_usable_track_id = 0
        self.logger = logger
        self.frame_num = 0
        self.trees = []
        self.reid_encoder = reid_encoder
        self.reid_feat_dim = 128 #self.reid_encoder.inference(image=np.zeros((100, 100, 3)), detections=np.array([[10, 20, 20, 30]])).shape[1]
        self.max_miss_count = self.cfg.get("max_miss_count", 20)
        self.max_tree_depth = max_tree_depth

        self.tree_timing = []
        self.prune_timing = []
        self.n_scan_pruning_timing = []

        # self.reid_score_mapping_file = REID_MAPPING_FILE
        # simil_off_diag = np.load(self.reid_score_mapping_file)
        # freq, bins = np.histogram(simil_off_diag, bins=np.linspace(0.9, 1.0, 50))
        # freq = freq / np.sum(freq)
        # cdf = np.cumsum(freq)

        bins = np.linspace(0.90, 1.0, num=11)
        if(name == "rider"):
            self.motion_gating_threshold = cfg.get("rider_motion_gating_threshold", 250)
            cdf = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.2, 0.7, 0.8, 0.9, 1.0])
            self.appearance_weight = cfg.get("rider_appearance_weight", 1)
            self.motion_weight = cfg.get("rider_motion_weight", 0)
        elif(name == "motor"):
            # Dont use the ReID model
            # TODO : Remove the ReID Encoder Inference as well
            self.motion_gating_threshold = cfg.get("motor_motion_gating_threshold", 250)
            cdf = np.array([0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.2, 0.7, 0.8, 0.9, 1.0]) # np.ones(11)
            # Give it some weight else the new dets will be pruned off
            self.appearance_weight = cfg.get("motor_appearance_weight", 0)
            self.motion_weight = cfg.get("motor_motion_weight", 1)
        self.min_reid_score = bins[0]
        self.reid_normalize_fn = interp1d(bins, cdf, kind='linear')
        self.tracker_dists = TrackerDists(self.name, self.reid_normalize_fn, self.min_reid_score, self.appearance_weight, self.motion_weight, self.motion_gating_threshold, self.miss_prob, self.max_miss_count, self.logger)

    def track(self, detections, frame, problem, reid_features=None):
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
        
        if(reid_features is None):
            reid_features = self.reid_encoder.inference(image=frame, detections=detections)
        # reid_features = np.random.randn(len(detections), 10)
        
        all_dets = []
        all_nodes = []
        for i, det in enumerate(detections):
            detection = Detection(det, reid_features[i]/np.linalg.norm(reid_features[i]), i)
            node = Node(self.frame_num, detection, i)
            all_nodes.append(node)
            all_dets.append(detection)
        
        # ### Return for viz
        # det_feats = []
        # for i in range(len(all_dets)):
        #     det_feats.append(all_dets[i].feature)

        # return det_feats
        
        # Make a null node for no detection
        node = Node(self.frame_num, Detection(np.zeros(5), np.zeros(self.reid_feat_dim), -1), -1)
        all_nodes.append(node)

        # Add the new nodes as the leaves of the existing trees
        count = 0
        t0 = time.time()
        for tree in self.trees:
            cost_prop = {"mot_score" : 0, "reid_score" : 0}
            tree.add_children_to_all_leaves_with_cost(all_nodes, self.tracker_dists.motion_conditional, self.tracker_dists.reid_conditional, cost_prop)
            count += 1
        self.tree_timing.append(time.time() - t0)
        
        print(self.name)
        for tree in self.trees:
            tree.print_tree()
        # Prune the newly formed tracks
        t0 = time.time()
        self.prune_tracks()
        self.prune_timing.append(time.time() - t0)
        
        print(self.name)
        for tree in self.trees:
            tree.print_tree()

        # Make new trees corresponding to the nodes
        for node in all_nodes:
            if(node.det_id != -1):
                self.trees.append(copy.deepcopy(node))

        if(self.frame_num % self.prune_after == 0):
            t0 = time.time()
            variables, all_paths, obj, all_costs = self.n_scan_pruning(problem, tracked=True)
            self.n_scan_pruning_timing.append(time.time() - t0)
        else:
            variables, all_paths, obj, all_costs = None, None, None, None

        # self.logger("###############################")
        # all_paths = self.give_all_paths()
        # self.logger(f"{self.frame_num} : ", len(all_paths))
        # self.logger("###############################")


        self.detections.append(all_dets)

        ### Return for viz
        det_feats = []
        for i in range(len(all_dets)):
            det_feats.append(all_dets[i].feature)

        return det_feats, variables, all_paths, obj, all_costs

    def give_tracks(self, tracks, assoc_ids):
        '''
        returns the final tracks obtained after solving the mwis problem
        '''
        self.logger(tracks)
        out = {}
        # self.logger(tracks)
        for i in range(self.frame_num):
            out[i + 1] = []

        for tid, track in tracks.items():
            for node in track:
                out[node.frame_num].append((tid, node.detection.bounding_box_xyxyc, assoc_ids[tid]))

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
    
    def give_all_costs(self, all_paths, tracked=False):
        '''
        gives all the costs for all the paths in the all the trees
        '''

        # Append all the paths of all the trees in a list
        all_motion_costs, all_reid_costs = [], []
        for tree in self.trees:
            if(tracked):
                if(tree.find_depth() <= 2):
                    continue
            motion_costs, reid_costs = tree.give_all_costs()
            all_motion_costs += motion_costs
            all_reid_costs += reid_costs


        all_motion_costs = np.array(all_motion_costs) + 0.10
        all_reid_costs = np.array(all_reid_costs) + 0.965
        all_costs = np.max(np.concatenate([all_reid_costs[None], all_motion_costs[None]]), 0)
        # all_costs = all_reid_costs

        for i, path in enumerate(all_paths):
            if((path[-1].mot_prune_score < 0) or (path[-1].cond_reid_score < 0)):
                all_costs[i] = -all_costs[i]
        
        print(all_costs)
        # all_costs = self.appearance_weight * all_reid_costs + self.motion_weight * all_motion_costs
        
        return all_costs
    
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
    
    def build_graph(self, paths, tracked=False):
        '''
        builds the graph given all the trees formed till now. For every path in every tree, make a node in the graph and if there is any
        detection common to any paths then there is an edge between the 2 nodes in the graph.
        '''
        
        all_paths = paths
        if(tracked):
            all_paths_tracked = []
            for i, path in enumerate(all_paths):
                if(len(path) <= 2):
                    continue
                all_paths_tracked.append(path)
            all_paths = all_paths_tracked

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
    
    def n_scan_pruning(self, problem, tracked=False):
        '''
        Do N-Scan pruning on all the trees. Solve the MWIS problem first and then any hypothesis starting from
        frame K - N (where K is the current frame number) which is diverging from the true solution will be pruned.
        '''
        self.logger("||||||||||||||||N SCAN PRUNING||||||||||||||||||")
        return self.solve_mwis_problem(problem, tracked=tracked)
    
    def after_n_scan(self, tracks, tracked=False):
        
        # From every tree, there can be only one possible solution, hence a track in tracks list wont occur in 2 trees at once
        tree_track_idx = []
        all_tree_paths = [] # If tracked, then all_tree_paths will contain the tracked paths only, else all paths
        all_tree_paths_untracked = []
        trees_tracked = []
        trees_untracked = []
        trees_single = []
        matched_detections_in_current_frame = [] # Contains the det_ids of the detections that got matched

        for i, tree in enumerate(self.trees):
            tree_path = tree.give_all_paths()
            if(tracked):
                depth = tree.find_depth()
                if(depth == 2):
                    all_tree_paths_untracked += tree_path
                    trees_untracked.append(tree)
                    continue
                elif(depth == 1):
                    trees_single.append(tree)
            all_tree_paths.append(tree_path)
            trees_tracked.append(tree)

        for track in tracks:
            for i, tree in enumerate(trees_tracked):
            # Check if the track belongs to this tree or not
                # print("Printing Tree")
                # tree.print_tree()
                if(tracked):
                    if(tree.find_depth() <= 2):
                        continue
                cond = False
                try:
                    path = all_tree_paths[i][0][::-1]
                except:
                    exit(0)
                    continue
                for j in range(min(len(path), len(track))):
                    if(path[j].det_id != -1 and path[j].det_id == track[j].det_id and path[j].frame_num == track[j].frame_num):
                        cond = True
                        break
                    elif(path[j].det_id != -1):
                        break
                if(cond):
                    # Track blongs to the tree, prune all the other paths belonging to the tree
                    tree.prune_except(track)
                    matched_detections_in_current_frame.append(track[-1].det_id)
                    tree_track_idx.append(i)
                    break

        # Remove all the trees that dont belong to any track
        rest_trees = []
        i = 0
        while(len(trees_tracked)):
            tree = trees_tracked.pop(0)
            if(i in tree_track_idx):
                rest_trees.append(tree)
            else:
                tree.prune_all_children()
            i += 1

        trees_new = []
        if(tracked):
            # Perform the second matching of the detections that are not assigned to any track in the first stage and the tracks that are untracked (len <= 2)
            frame_det_ids = [det.det_id for det in self.detections[-1]]
            u_dets = [det_id for det_id in frame_det_ids if det_id not in matched_detections_in_current_frame]
            if(len(u_dets) != 0):
                paths_filtered = []
                if(len(all_tree_paths_untracked) == 0):
                    for i in range(len(u_dets)):
                        trees_new.append(trees_single[u_dets[i]])
                else:
                    for path in all_tree_paths_untracked:
                        print(path)
                        if(path[0].det_id in u_dets): # path is from bottom to top
                            paths_filtered.append(path)
                    cost_matrix = np.array([path[0].cond_mot_score for path in paths_filtered]).reshape(len(paths_filtered)//len(u_dets), len(u_dets))
                    # cost_matrix[cost_matrix < 0.1] = 0
                    print(len(paths_filtered))
                    print(1-cost_matrix)
                    _, x, y = lap.lapjv(1-cost_matrix, extend_cost=True, cost_limit=0.9)
                    matched_x = [ix for ix, mx in enumerate(x) if mx >= 0]
                    print("LAP ", matched_x)
                    # unmatched_a = np.where(x < 0)[0]
                    unmatched_y = np.where(y < 0)[0]
                    for i in range(len(paths_filtered)//len(u_dets)):
                        if(i in matched_x):
                            matched_det_idx = x[i]
                            track = paths_filtered[i * len(u_dets) + matched_det_idx][::-1]
                            trees_untracked[i].prune_except(track)

                            # Assign track id to track
                            trees_untracked[i].track_id = self.next_usable_track_id
                            trees_untracked[i].children[0].track_id = self.next_usable_track_id
                            self.next_usable_track_id += 1

                            trees_new.append(trees_untracked[i])
                    
                    for i in unmatched_y:
                        trees_new.append(trees_single[u_dets[i]])
            

        
        self.trees = rest_trees + trees_new
        # Prune trees that have miss count more than self.max_miss_count
        rest_trees = []
        i = 0
        while(len(self.trees)):
            tree = self.trees.pop(0)
            try:
                tree_path = tree.give_all_paths()[0]
            except:
                continue
            miss_count = 0
            max_miss_count = 0
            for node in tree_path:
                if(node.det_id == -1):
                    miss_count += 1
                else:
                    max_miss_count = max(max_miss_count, miss_count)
                    miss_count = 0
            max_miss_count = max(max_miss_count, miss_count)
            if(max_miss_count < self.max_miss_count):
                rest_trees.append(tree)
            else:
                tree.prune_all_children()
            i += 1
        
        self.trees = rest_trees

        
        # After pruning each tree is essentially a linear array, limit the size of the array
        if(self.max_tree_depth != -1):
            for i, tree in enumerate(self.trees):
                if(tree.find_depth() == self.max_tree_depth + 1):
                    curr_tree = tree
                    child_tree = tree.children[0]
                    self.trees[i] = child_tree
                    del curr_tree
        
        # Update the running reid feature vector of the tree
        for i, tree in enumerate(self.trees):
            tree.update_ema_feat()

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

    def solve_mwis_problem(self, problem, tracked=False):
        '''
        build the mwis problem and solve it
        returns the tracks formed
        '''
        
        all_paths, adj_mat = self.build_graph(paths=self.give_all_paths(), tracked=tracked)
        all_costs = self.give_all_costs(all_paths, tracked=tracked)

        # self.logger("**********MWIS**********")
        # self.logger("ALL PATHS")
        # print("ALL_PATHS")
        # print(self.name)
        # for i, path in enumerate(all_paths):
        #     print(path)
        #     print(all_costs[i])
        #     self.logger(path)
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
            tree.prune_paths([], self.tracker_dists.prune_condition_function_cached)

    def null(self, *args, **kwargs):
        pass
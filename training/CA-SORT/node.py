import numpy as np
import copy
from utils.kalman_filter import KalmanFilterXYWH

class Node():
    def __init__(self, frame_num, det, det_id):
        self.frame_num = frame_num
        self.det_id = det_id
        self.mot_score = 1e-3
        self.reid_score = 1e-3
        self.cond_mot_score = 1e-3
        self.cond_reid_score = 1e-3
        self.mot_prune_score = 0
        self.detection = det
        self.kf_mean, self.kf_cov = KalmanFilterXYWH().initiate(self.detection.bounding_box_xywhc[:4])
        self.ema_coeff = 0.9
        self.running_reid_feat = None
        self.children = None
        self.track_id = -1

    def update_ema_feat(self, curr_feat=None):
        if(self.children == None):
            if(curr_feat is None):
                self.running_reid_feat = self.detection.feature
                return
            if(self.det_id != -1):
                self.running_reid_feat = self.ema_coeff * self.detection.feature + (1 - self.ema_coeff) * curr_feat
                self.running_reid_feat = self.running_reid_feat / np.linalg.norm(self.running_reid_feat)
            else:
                self.running_reid_feat = curr_feat
            return

        for child in self.children:
            child.update_ema_feat(curr_feat=self.running_reid_feat)

    def find_depth(self):
        if(self.children == None):
            return 1

        max_depth = 0
        for child in self.children:
            depth = child.find_depth()
            max_depth = max(depth, max_depth)
        
        return max_depth + 1

    def add_children_to_all_leaves(self, children):
        '''
        add children (a list of Nodes) to all the leaves of the tree starting from this node
        '''
        if(self.children == None):
            # Creating a copy is necessary!
            self.children = copy.deepcopy(children)
            return
        
        for i in range(len(self.children)):
            self.children[i].add_children_to_all_leaves(children)
    
    def add_children_to_all_leaves_with_cost(self, children, motion_cost_fn, reid_cost_fn, cost_prop={}, ancestors=[]):
        '''
        add children (a list of Nodes) to all the leaves of the tree starting from this node
        '''
        ancestors.append(self)
        if(self.children == None):
            # Creating a copy is necessary!
            self.children = copy.deepcopy(children)
            for child in self.children:
                ancestors.append(child)
                mean, cov = self.kf_mean, self.kf_cov
                print(ancestors)
                child.cond_mot_score, child.mot_prune_score, mean, cov = motion_cost_fn(mean, cov, child.detection, self.det_id)
                child.cond_reid_score = reid_cost_fn(ancestors)
                child.mot_score = child.cond_mot_score + self.mot_score
                child.reid_score = child.cond_reid_score + self.reid_score
                child.kf_mean, child.kf_cov = mean, cov
                child.track_id = self.track_id
                print(child.cond_mot_score)
                print(child.mot_prune_score)
                print(child.cond_reid_score)
                ancestors.pop()
            ancestors.pop()
            return
        
        # cost_prop_new = {}
        # cost_prop_new['kf_mean'] = self.kf_mean
        # cost_prop_new['kf_cov'] = self.kf_cov
        # cost_prop_new['mot_score'] = cost_prop['mot_score'] + self.mot_score
        # cost_prop_new['reid_score'] = cost_prop['reid_score'] + self.reid_score

        for i in range(len(self.children)):
            self.children[i].add_children_to_all_leaves_with_cost(children, motion_cost_fn, reid_cost_fn, ancestors=ancestors)
        ancestors.pop()

    def __repr__(self):
        return f"{self.frame_num}-{self.det_id}"
    
    def give_all_paths(self):
        '''
        returns a list of all the paths in the tree
        '''
        if(self.children == None):
            return [[self]]

        path_list = []
        for child in self.children:
            child_path = child.give_all_paths()
            # Append the current node to all the paths in the child_path
            for path in child_path:
                path.append(self)
            path_list += child_path

        return path_list
    
    def give_all_costs(self):
        '''
        returns a list of the costs of all the paths in the tree
        '''
        if(self.children == None):
            # return [self.mot_score], [self.reid_score]
            return [self.cond_mot_score], [self.cond_reid_score]

        motion_cost = []
        reid_cost = []
        for child in self.children:
            child_mot_scores, child_reid_scores = child.give_all_costs()
            motion_cost += child_mot_scores
            reid_cost += child_reid_scores

        return motion_cost, reid_cost
    
    def prune_paths(self, path, prune_fn):
        '''
        path is the path of nodes till now
        prune_fn returns 1 if the path is to be pruned, else 0
        returns a list of all the paths in the tree
        '''

        if(self.children == None):
            to_prune = prune_fn(path + [self])
            return to_prune # Return to our parent

        i = 0
        while(i < len(self.children)):
            ret = self.children[i].prune_paths(path + [self], prune_fn)
            if(ret >= 1):
                # print(f"Removing {self.children[i].det_id} as child of {self.det_id}")
                del self.children[i] # Remove the node from the list of children, hence dissolving the path
                i -= 1
                # print(self.children)
            i += 1

        # Dont return anything meaningful to our parent
        return -1
    
    def prune_all_children(self):
        '''
        recursively prunes all the children of the tree node
        '''
        if(self.children == None):
            return

        while(len(self.children)):
            self.children[0].prune_all_children()
            del self.children[0] # Remove the node from the list of children, hence dissolving the path
    
    def prune_except(self, track):
        '''
        path is the path of nodes
        removes all tracks except this one
        '''

        if(self.det_id == track[0].det_id and self.frame_num == track[0].frame_num):
            if(self.children == None):
                return True # Return to our parent
            i = 0
            while(i < len(self.children)):
                ret = self.children[i].prune_except(track[1:])
                if not ret:
                    # print(f"Pruning {self.children[i]} child from parent {self}")
                    del self.children[i]
                    i -= 1
                i += 1
            return True
        else:
            self.prune_all_children()
            return False # Indicate to our parent that we want to delete this child

    def print_tree(self, ancestors=[]):
        ancestors.append(self)
        if(self.children == None):
            print(ancestors)
            ancestors.pop()
            return

        for i in range(len(self.children)):
            self.children[i].print_tree(ancestors=ancestors)
        ancestors.pop()
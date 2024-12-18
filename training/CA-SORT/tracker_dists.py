import numpy as np
from utils.kalman_filter import KalmanFilterXYWH
import torch

class TrackerDists():

    def __init__(self, name, reid_normalize_fn, min_reid_score, appearance_weight, motion_weight, motion_gating_threshold, miss_prob, max_miss_count, logger):
        self.name = name
        self.kalman_filter = KalmanFilterXYWH()
        self.reid_normalize_fn = reid_normalize_fn
        self.min_reid_score = min_reid_score
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        self.max_miss_count = max_miss_count
        self.motion_gating_threshold = motion_gating_threshold
        self.logger = print #self.null
        self.miss_prob = miss_prob

    def prune_condition_function(self, path):
        '''
        condition for pruning the tracks (returns 1 if track is to be pruned else 0)
        path is the path from the root to the leaf node (a list of nodes)
        '''
        if(path[-1].det_id == -1):
            return 0

        self.logger(path)
        dist_appearance_score, dist_appearance_mat = self.compute_appearance_dist_for_track(path)
        dist_motion_probs = self.compute_motion_dist_for_track(path)
        dist_motion_probs = dist_motion_probs[dist_motion_probs != self.miss_prob]

        gating_cond = self.appearance_weight * (dist_appearance_score < 0) + self.motion_weight * (dist_motion_probs[-1] < 0).item()
        self.logger(f"PATH : {path} : Dist Appearance Score {dist_appearance_score} : {gating_cond}")
        self.logger()

        if(gating_cond):
            return 1
        else:
            return 0
        
    def prune_condition_function_cached(self, path):
        '''
        condition for pruning the tracks (returns 1 if track is to be pruned else 0)
        path is the path from the root to the leaf node (a list of nodes)
        '''
        
        if(path[-1].det_id == -1):
            self.logger(f"PATH : {path} : Dist Cond Appearance Score {path[-1].cond_reid_score}, MOT_PRUNE_SCORE {path[-1].mot_prune_score}, Dist Cond Motion Score {path[-1].cond_mot_score}: {0}")
            return 0

        if(len(path) <= 2):
            return 0

        self.logger(path)
        gating_cond = (self.appearance_weight == 1) * (path[-1].cond_reid_score < 0)
        gating_cond = (path[-1].cond_reid_score < 0)
        gating_cond += (path[-1].mot_prune_score < 0)

        # gating_cond = 1

        self.logger(f"PATH : {path} : Dist Cond Appearance Score {path[-1].cond_reid_score}, MOT_PRUNE_SCORE {path[-1].mot_prune_score}, Dist Cond Motion Score {path[-1].cond_mot_score}: {gating_cond}")
        self.logger()

        if(gating_cond):
            return 1
        else:
            return 0
        
    def compute_dist_for_track(self, track):
        '''
        track is a list of length T representing the detections corresponding to the hypothesis track throughout the T frames
        
        returns the gating distance for the track hypothesis (acts like log(prob) for the track's kinematic component)
        '''

        self.logger(track)
        motion_dist_probs = self.compute_motion_dist_for_track(track)

        motion_dist_probs = motion_dist_probs[motion_dist_probs != self.miss_prob]

        # TODO:Change this 
        # Remember we want to focus on the change of the function, whatever function we define doesnt matter, and its magnitude doesnt matter,
        # sm(t + 1) - sm(t) > 0 if we support the hypothesis, else < 0
        motion_dist = motion_dist_probs.sum() # 0 if motion_dist_probs[-1] >= 0 else 0

        app_score, dists = self.compute_appearance_dist_for_track(track)

        a = self.appearance_weight
        b = self.motion_weight
        dist = app_score * a + (motion_dist) * b
        self.logger(track)
        self.logger(dist)
        return dist

    def compute_motion_dist_for_track(self, track):
        '''
        track is a list of length T representing the detections corresponding to the hypothesis track throughout the T frames
        
        returns the gating distance for the track hypothesis (acts like log(prob) for the track's kinematic component)
        '''
        
        mean, covariance = self.kalman_filter.initiate(track[0].detection.bounding_box_xywhc[:4])
        # Remember that bounding boxes that are all zeros are null detections (occlusion)
        probs = np.ones(len(track)) * self.miss_prob
        probs[0] = 0
        gating_threshold = self.motion_gating_threshold
        for t in range(1, len(track)):
            mean, covariance = self.kalman_filter.predict(mean, covariance)
            # self.logger("COVARIANCE :", t, covariance)
            curr_bb = track[t].detection.bounding_box_xywhc[:4]
            iou_dist = self.iou_dist(mean[:4], curr_bb)
            print("IOU DIST : ", iou_dist)
            probs[t] = iou_dist - 0.2
            # Find the gating distance w.r.t the current detection and the mean and the covariance till now
            if(track[t].det_id != -1):
                maha_dist_sq = self.kalman_filter.gating_distance(mean, covariance, track[t].detection.bounding_box_xywhc[None, :4], only_position=False)
                # self.logger("DET : ", 1 / np.linalg.det(covariance))
                self.logger("MOTION PROB : ", np.exp(- 0.5 * np.log(np.linalg.det(covariance[:2, :2])) - 0.5 * maha_dist_sq))
                self.logger("MAHA : ", maha_dist_sq)
                if(track[t-1].det_id != -1):
                    gating_threshold = 20
                probs[t] = np.exp(- 0.5 * np.log(np.linalg.det(covariance[:2, :2])) - 0.5 * maha_dist_sq) - 1e-6
                mean, covariance = self.kalman_filter.update(mean=mean, covariance=covariance, measurement=track[t].detection.bounding_box_xywhc[:4])
                gating_threshold = self.motion_gating_threshold
        print()
        return probs

    def motion_conditional(self, mean, covariance, detection, det_id):
        '''
        mean is a 8 dim vector, covariance is the 8*8 matrix of the tracklet hypothesis,
        detection is the Detection object representing the current detection 

        returns the score for the tracklet and the updated mean and covariance
        '''
        # If the current track is a lost tracked, then the velocities in the width and height should be ignored
        if(det_id == -1):
            mean[6] = 0
            mean[7] = 0
        mean, covariance = self.kalman_filter.predict(mean, covariance)
        if(detection.det_id != -1):
            curr_bb = detection.bounding_box_xywhc[:4]
            iou_dist = self.iou_dist(mean[:4], curr_bb)
                
            maha_dist_sq = self.kalman_filter.gating_distance(mean, covariance, detection.bounding_box_xywhc[None, :4], only_position=False)
            self.logger("MOTION PROB : ", np.exp(- 0.5 * np.log(np.linalg.det(covariance)) - 0.5 * maha_dist_sq))
            self.logger("MAHA : ", maha_dist_sq)
            self.logger("MOTION DIST : ", - 0.5 * np.log(np.linalg.det(covariance)) - 0.5 * maha_dist_sq)
            mot_prune_score = (- 0.5 * np.log(np.linalg.det(covariance)) - 0.5 * maha_dist_sq + self.motion_gating_threshold).item()
            if(iou_dist == 0):
                # A negative number between -1 and 0
                mot_score_cond = 2e-3#(np.exp(mot_prune_score - self.motion_gating_threshold) - 0.5).item()
            else:
                mot_score_cond = iou_dist - 0.10

            mot_score_cond = mot_prune_score = iou_dist - 0.10
            
            mean, covariance = self.kalman_filter.update(mean=mean, covariance=covariance, measurement=detection.bounding_box_xywhc[:4])
        else:
            mot_prune_score = 1e-3
            mot_score_cond = 1e-3
        print("MOTION CONDITIONAL")
        print("MOTION COND SCORE : ", mot_score_cond)
        print("MOTION PRUNE SCORE : ", mot_prune_score)
        print()
        return mot_score_cond, mot_prune_score, mean, covariance

    def compute_appearance_dist_for_track(self, track):
        '''
        track is a list of length T representing the nodes of the detections corresponding to the hypothesis track throughout the T frames
        
        returns the reid cost for the track hypothesis (is the log(prob_track/prob_null))
        '''
        score = 0
        for i in range(len(track)):
            score += self.reid_conditional(track[:i+1])

        features = []
        for t in range(len(track)):
            if(track[t].det_id != -1):
                features.append(track[t].detection.feature)
        
        features = np.array(features) # (T, D)
        dists = features @ features.T # (T, T)
        dists[dists > 1] = 1
        dists[dists < self.min_reid_score] = self.min_reid_score
        dists = self.reid_normalize_fn(dists)

        return score, dists
    
    def reid_conditional(self, track):
        if(len(track) == 1):
            # Be aware of the tolerance of the solver (default 1e-4 for Gurobi), should be bigger than that
            return 1e-3
        
        num_miss = 0
        for t in track:
            if(t.det_id == -1): 
                num_miss += 1
        
        if(num_miss == len(track)):
            return -1e-4

        if(track[-1].det_id == -1):
            return 1e-3

        score = track[-1].detection.feature @ track[-2].running_reid_feat
        # try:
        #     score = self.reid_normalize_fn(score).item()
        # except:
        #     score = 0
        score -= 0.965
        print("REID CONDITIONAL")
        print(track)
        print("REID SCORE : ", score)
        print()
        return score #(-np.log(0.50) + np.log(probability)).item()
    
    def iou_dist(self, box1, box2):

        x1, y1, w1, h1 = box1
        x1, x2 = x1 - w1/2, x1 + w1/2
        y1, y2 = y1 - h1/2, y1 + h1/2

        x3, y3, w3, h3 = box2
        x3, x4 = x3 - w3/2, x3 + w3/2
        y3, y4 = y3 - h3/2, y3 + h3/2

        intersection_width = max(0, min(x2, x4) - max(x1, x3))
        intersection_height = max(0, min(y2, y4) - max(y1, y3))
        area_intersection = intersection_width * intersection_height

        area_box1 = (x2 - x1) * (y2 - y1)
        area_box2 = (x4 - x3) * (y4 - y3)

        area_union = area_box1 + area_box2 - area_intersection

        iou = area_intersection / max(area_union, 1e-10)

        return iou
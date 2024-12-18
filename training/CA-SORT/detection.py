import numpy as np

class Detection():
    def __init__(self, bounding_box, appearance_feature, det_id):
        self.bounding_box_xyxyc = bounding_box.astype(np.int32)
        self.bounding_box_xywhc = self.xyxyc_to_xywhc()
        self.xcenter = bounding_box[0]
        self.ycenter = bounding_box[1]
        self.width = bounding_box[2]
        self.height = bounding_box[3]
        self.conf = bounding_box[4]
        self.feature = appearance_feature
        self.det_id = det_id
    
    def xywhc_to_xyxyc(self):
        x, y, w, h, c = self.bounding_box_xywhc

        return np.array([x-w/2, y-h/2, x+w/2, y+h/2, c]).astype(np.int32)

    def xyxyc_to_xywhc(self):
        xmin, ymin, xmax, ymax, c = self.bounding_box_xyxyc

        return np.array([(xmin + xmax)/2, (ymin + ymax)/2, xmax - xmin, ymax - ymin, c]).astype(np.int32)
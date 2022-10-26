"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import time
# from filterpy.kalman import KalmanFilter
from .kalman_filter import KalmanFilter
from .lp.yolov5 import YoLov5TRT
from .lp.recognitionpth import Recognition
# from .lp.recognition import Recognition
import numpy as np
import cv2
import threading
# import numba
from scipy.optimize import linear_sum_assignment

import rospy

# np.random.seed(0)
OBJ_LIST = ['Car', 'Misc', 'Truck', 'rcircle', 'Cement_truck',
            'Pedestrian', 'Cyclist', 'gcircle', 'TrafficLight_Black',
            'Bus', 'ycircle', 'rperson', 'Tram', 'gperson', 'Trailer',
            'TrafficLight_Red', 'gup', 'Fule_Tank', 'TrafficLight_Green',
            'rup', 'yright', 'rright', 'rleft', 'TrafficLight_Dig',
            'gleft', 'Dump_Truck', 'TrafficLight_Yellow', 'gbike']
CAR_LIST = ['Car', 'Truck', 'Cement_truck',
            'Bus', 'Fule_Tank', 'Dump_Truck', 'Trailer']

# @numba.jit
def linear_assignment(cost_matrix):
    # # try:
    # _, x, y = lapjv(cost_matrix, extend_cost=True)
    # return np.array([[y[i], i] for i in x if i >= 0])
    # # except ImportError:
    #     
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# @numba.njit
def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None, clss=None, lp=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score, clss, lp], dtype=object).reshape((1, 7))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], 
                            [0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], 
                            [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], 
                             [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.conf = bbox[4]  # conf
        self.cls = bbox[5]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # lp
        self.lp = 'UnKnown'
        self.car_img = None
        
        # self.avg = 0
            
    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        # self.history = []
        self.hits += 1
        self.hit_streak += 1
        # start_update = time.time()
        self.kf.update(convert_bbox_to_z(bbox))  # input: (x,y,s,r)^T
        # time_update = time.time() - start_update
        # self.avg = time_update if self.avg==0 else self.avg*0.95+time_update*0.05
        # print('track_kf_update : {} ms'.format(self.avg * 1000))
        

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # self.history.append(convert_x_to_bbox(self.kf.x))
        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x, self.conf, self.cls, self.lp)  # [x1,y1,x2,y2,conf,cls,lp]


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        # test time
        self.avg1 = 0
        self.avg2 = 0
        self.avg3 = 0
        self.avg4 = 0
        # lp 
        self.det = YoLov5TRT()
        self.ocr = Recognition()
        self.tasks = []
        self.lock = threading.Lock()
        self.thread2 = threading.Thread(target = self.lp_job)
        
    def lp_job(self):
        while True:  
            # print('thread runing')
            if len(self.tasks)>0:
                self.lock.acquire()
                current_trk, current_car = self.tasks.pop(0) 
                self.lock.release()
                lp_box = self.det.infer([current_car])
                print(lp_box)
                if lp_box is not None:
                    current_lp = current_car[lp_box[1]:lp_box[3], lp_box[0]:lp_box[2]] 
                    lp = self.ocr.infer([current_lp])
                    current_trk.lp = lp
                    print('lp = {}'.format(lp[0]))
                else:
                    lp = self.ocr.infer2([current_car])
                    current_trk.lp = lp
            time.sleep(1)
            

    def update(self, frame_img, dets=np.empty((0, 6))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 7)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if self.frame_count<=0:
            self.thread2.start()
        # print('sort runing')
            
        self.frame_count += 1
        # get predicted locations from existing trackers.
        start1 = time.time()
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]              # [x,y,x,r]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]   
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        time1 = time.time() - start1          # kf predict
        start2 = time.time()
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)
        time2 = time.time() - start2          # associate
        # update matched trackers with assigned detections
        # m[0]是检测器ID， m[1]是跟踪器ID
        
        start3 = time.time()
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :]) 
        time3 = time.time() - start3    # kal init
        start5 = time.time()
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk) 
            x1, y1, x2, y2 = int(dets[i,0]), int(dets[i,1]), int(dets[i,2]), int(dets[i,3])
            if inroi(x1, y1, x2, y2) and (OBJ_LIST[int(dets[i, 5])] in CAR_LIST):
                car_img = frame_img[y1:y2, x1:x2]
                self.lock.acquire()
                self.tasks.append((trk,car_img))
                self.lock.release()
        
        time5 = time.time() - start5
        i = len(self.trackers) 
        
        start4 = time.time()
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        time4 = time.time() - start4 
        self.avg1 = time1 if self.avg1==0 else self.avg1 * 0.95 + time1 * 0.05   
        self.avg2 = time2 if self.avg2==0 else self.avg2 * 0.95 + time2 * 0.05   
        self.avg3 = time3 if self.avg3==0 else self.avg3 * 0.95 + time3 * 0.05   
        self.avg4 = time4 if self.avg4==0 else self.avg4 * 0.95 + time4 * 0.05   
        # print('kf predict              : {:.2f} ms'.format(self.avg1 * 1000))
        # print('associate               : {:.2f} ms'.format(self.avg2 * 1000))
        # print('kf update               : {:.2f} ms'.format(self.avg3 * 1000))
        # print('pop                     : {:.2f} ms'.format(self.avg4 * 1000))
        # print('lp                      : {:.2f} ms'.format(time5 * 1000))
        if (len(ret) > 0):
            return np.concatenate(ret)
        
        return np.empty((0, 8))
    
    
    
def inroi(x1, y1, x2, y2):
    if y1+y2 > 1110 and x1+x2 > 2140:
        return True
    else:
        return False

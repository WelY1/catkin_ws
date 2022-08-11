from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

import time

from ..yolov5.lpdetector import Detector
from ..lp.recognition import Recognition

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
        测量与轨迹关联的距离度量
    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹前的最大未命中数
    n_init : int
        Number of frames that a track remains in initialization phase.
        确认轨迹前的连续检测次数。如果前n_init帧内发生未命中，则将轨迹状态设置为Deleted
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter() # 实例化卡尔曼滤波器
        self.tracks = []   # 保存一个轨迹列表，用于保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id
        
        self.det = Detector()
        self.ocr = Recognition()
        

 
    def predict(self):
        """Propagate track state distributions one time step forward.
        将跟踪状态分布向前传播一步

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)          # track.predict

    
    def inROI(self, x1, y1, x2, y2, h, w):
        
        pass
    
    def update(self, ori_img, detections):
        """Perform measurement update and track management.
        执行测量更新和轨迹管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        
        update_start = time.time()
        # 1. 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # 更新tracks中相应的detection
            target = detections[detection_idx]
            self.tracks[track_idx].update(self.kf, target) # 更新位置和track的状态
            # 如果车辆车牌没被锁定就更新车牌
            if not len(self.tracks[track_idx].lp_confirmed):
                x1, y1, x2, y2 = target.car[0], target.car[1], target.car[2], target.car[3]
                # 如果车在ROI范围内就识别车牌
                # if self.inROI(x1, y1, x2, y2, ori_img.shape[0], ori_img.shape[1]):
                car_img = ori_img[y1:y2, x1:x2]
                lpbox = self.det.detect(car_img)
                if len(lpbox):
                    x1, y1, x2, y2 = lpbox[0], lpbox[1], lpbox[2], lpbox[3]
                    im_crops = []
                    im_crops.append(car_img[y1:y2,x1:x2])
                    lp, conf = self.ocr(im_crops)
                    if len(lp):
                        self.tracks[track_idx].lp_queue(lp)
        
        update_time = time.time() - update_start
        # print(update_time)
               
        # 2. 针对未匹配的track, 调用mark_missed进行标记
        # track失配时，若Tantative则删除；若update时间很久也删除
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. 针对未匹配的detection， detection失配，进行初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # 得到最新的tracks列表，保存的是标记为Confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # self.tracks_id = [t.track_id for t in self.tracks]  # 保存的是tracks中每个track的id用来做映射

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]   # 不包括tentative的track
        features, targets = [], []
        for track in self.tracks:
            # 获取所有Confirmed状态的track id
            if not track.is_confirmed():
                continue
            features += track.features # 将Confirmed状态的track的features添加到features列表
            # 获取每个feature对应的trackid
            targets += [track.track_id for _ in track.features]
            track.features = []
        # 距离度量中的特征集更新
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])           # 所有检测对象的特征
            targets = np.array([tracks[i].track_id for i in track_indices])                 # 跟踪对象的id
            cost_matrix = self.metric.distance(features, targets)                             # 通过最近邻（余弦距离）计算出成本矩阵（代价矩阵）
            #self.metric 是NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)对象
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)                                                  # 判断距离关系，使用马氏距离，大于阈值的都把代价变成很大的一个数

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 区分开confirmed tracks和unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]         #  获取状态非confirmed的对象，状态为Tentative，Deleted

        # Associate confirmed tracks using appearance features.
        # 对确定态的轨迹进行级联匹配，得到匹配的tracks、不匹配的tracks、不匹配的detections
        # matching_cascade 根据特征将检测框匹配到确认的轨迹。
        # 传入门控后的成本矩阵
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # 将未确定态的轨迹和刚刚没有匹配上的轨迹组合为 iou_track_candidates 
        # 并进行基于IoU的匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1] # 刚刚没有匹配上的轨迹
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1] # 并非刚刚没有匹配上的轨迹
        # 对级联匹配中还没有匹配成功的目标再进行IoU匹配
        # min_cost_matching 使用匈牙利算法解决线性分配问题。
        # 传入 iou_cost，尝试关联剩余的轨迹与未确认的轨迹。
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b # 组合两部分匹配 
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())       # (center x, center y, aspect ratio, height) 估计的是box中心的位置

        # lp = ''
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, 
            detection.feature))
            
        self._next_id += 1

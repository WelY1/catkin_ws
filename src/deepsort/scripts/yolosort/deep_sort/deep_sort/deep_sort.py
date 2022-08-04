import numpy as np
import torch

from rospkg import RosPack

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort'] # __all__ 提供了暴露接口用的”白名单“


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, 
                    nms_max_overlap=1.0, max_iou_distance=0.7, 
                    max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence # 检测结果置信度阈值 
        self.nms_max_overlap = nms_max_overlap # 非极大抑制阈值，设置为1代表不进行抑制
        
        model_path = RosPack().get_path('deepsort') + '/scripts/yolosort/' + model_path
        self.extractor = Extractor(model_path, use_cuda=use_cuda) # 用于提取一个batch图片对应的特征

        max_cosine_distance = max_dist # 最大余弦距离，用于级联匹配，如果大于该阈值，则忽略
        nn_budget = 100 # 每个类别gallery最多的外观描述子的个数，如果超过，删除旧的
        # NearestNeighborDistanceMetric 最近邻距离度量
        # 对于每个目标，返回到目前为止已观察到的任何样本的最近距离（欧式或余弦）。
        # 由距离度量方法构造一个 Tracker。
        # 第一个参数可选'cosine' or 'euclidean'
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)           # Matching cascade初始化
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)         # IOU匹配初始化
        

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        # 从原图中抠取bbox对应图片并计算得到相应的特征
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh) 
        # detetction.car存的是car在图中的坐标
        cars = self._get_cars(bbox_xywh)   
        '''
        TODO:
            第一种思路是在这里（匹配前）计算detection的embedding的时候就把车牌检测并OCR，存到detection中
            缺点：车牌信息对匹配的作用不大，且计算耗时间，当前帧发出去的消息需要等很多帧才能收到，关键是detection只记录当前帧的信息，不记录历史消息，因此不考虑
            伏笔：提取embedding特征的时候可以把特征层用于车牌检测，节省一步时间
            这步也可以不在这里就扣图，detection里存坐标也可以，如果超显存了，可以试试之后检测的时候以其扣图
        '''
        
        # 筛选掉小于min_confidence的目标，并构造一个Detection对象构成的列表
        detections = [Detection(bbox_tlwh[i], conf, features[i], cars[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]   # 转换数据格式
        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)          # 返回nms后剩下的bbox的索引值[]
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict() # 将跟踪状态分布向前传播一步
        self.tracker.update(ori_img, detections) # 执行测量更新和跟踪管理

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_lp = track.lp_confirmed      # lp = ''
            
            outputs.append([x1,y1,x2,y2,track_id,track_lp])   
            # outputs.append(np.array([x1,y1,x2,y2,track_id,track_lp], dtype=np.int))   # 感觉这里有问题，可能lp得编码一下
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    #将bbox 中心的[x,y,w,h] 转换成 左上角[x,y,w,h]
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    #将bbox的[x,y,w,h] 转换成[x1,y1,x2,y2]
    #某些数据集例如 pascal_voc 的标注方式是采用[x，y，w，h]
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    # 获取抠图部分的特征
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2] # 抠图部分
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops) # 对抠图部分提取特征
        else:
            features = np.array([])
        return features
        
    def _get_cars(self, bbox_xywh):
        car_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            # im = ori_img[y1:y2,x1:x2] # 抠图部分
            car_crops.append((x1,y1,x2,y2))
        return car_crops



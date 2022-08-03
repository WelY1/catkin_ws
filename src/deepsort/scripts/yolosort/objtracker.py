from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np

import time

from rospkg import RosPack

cfg = get_config()
cfg.merge_from_file(RosPack().get_path('deepsort') + "/scripts/yolosort/deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, 
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, 
                    n_init=cfg.DEEPSORT.N_INIT, 
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)                                                                          # deepsort 实例化
character = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ藏川鄂甘赣港贵桂黑沪吉冀津晋京辽鲁蒙闽宁青琼陕苏皖湘新渝豫粤云浙'

def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4
    
    for (x1, y1, x2, y2, pos_id,lp) in bboxes:
        color = (0, 255, 0)
            
        # check whether hit line                                                  每个bbox的标记点，可以用来计数
        # check_point_x = x1                                                         # check_point_x   
        # check_point_y = int(y1 + ((y2 - y1) * 0.6))              # check_point_y 
        check_point_x = int(x1 + ((x2 - x1) * 0.5))              
        check_point_y = int(y1 + ((y2 - y1) * 0.5))
        

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)          # detection box plot
        tf = max(tl - 1, 1)  # font thickness
        c2 = c1[0], c1[1] - 3
        # if len(lp_index):
        #     lplist = [character[c] for c in lp_index]
        #     lp = ''.join(lplist)[1:]
        # else:
        #     lp = 'None'
        cv2.putText(image, 'CAR-{}-{}'.format(lp,pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    
        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

        ndarray_pts = np.array(list_pts, np.int32)
        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        list_pts.clear()
    return image

def update(target_detector, image):

        det_start = time.time()
        _, bboxes = target_detector.detect(image)              #  img,  [x1, y1, x2, y2,  label, confidence, lp_img]
        det_time = time.time() - det_start
        
        bbox_xywh = []
        confs = []
        bboxes2draw = []
        if len(bboxes):
            # Adapt detections to deep sort input format
            for x, y, w, h, _, conf in bboxes:
                # [center_x, center_y, w, h]
                bbox_xywh.append([x, y, w, h])                                           
                confs.append(conf)                             # [confidence]
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
    
            dpsort_start = time.time()
            # Pass detections to deepsort
            outputs = deepsort.update(xywhs, confss, image)       # ipnut:bbox_xywh, confidences, ori_img      output: [x1,y1,x2,y2,track_id,track_lp]
            dpsort_time = time.time() - dpsort_start
            # print('det:{} fps, dpsort:{} fps'.format(int(1/det_time),int(1/dpsort_time)))
            
            
            for value in list(outputs):
                x1,y1,x2,y2,track_id,track_lp = value
                x1 = eval(x1)
                x2 = eval(x2)
                y1 = eval(y1)
                y2 = eval(y2)
                track_id = eval(track_id)
                bboxes2draw.append((x1, y1, x2, y2, track_id, track_lp))
                # print((x1+x2)/2, (y1+y2)/2, track_id, track_lp)
        image = plot_bboxes(image, bboxes2draw)

        return image, bboxes2draw

import torch
import time
import numpy as np
import cv2

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import yolov4.darknet as darknet

import objtracker

from rospkg import RosPack

'''
OBJ_LIST = ['Car', 'Misc', 'Truck', 'rcircle', 'Cement_truck', 
'Pedestrian', 'Cyclist', 'gcircle', 'TrafficLight_Black', 'Bus', 
'ycircle', 'rperson', 'Tram', 'gperson', 'Trailer', 'TrafficLight_Red',
'gup', 'Fule_Tank', 'TrafficLight_Green', 'rup', 'yright', 'rright', 'rleft',
'TrafficLight_Dig', 'gleft', 'Dump_Truck', 'TrafficLight_Yellow', 'gbike']
'''

class baseDet(object):
    def __init__(self):
        self.thresh = 0.25
        self.stride = 1

    def build_config(self):
        self.frameCounter = 0

    def feedCap(self, im, func_status):
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }
        self.frameCounter += 1
        
        start = time.time()
        im, obj_bboxes = objtracker.update(self, im)
        frame_time = time.time() - start
        # print('******frame: {} fps*********'.format(int(1/frame_time)))
        
        retDict['frame'] = im
        retDict['obj_bboxes'] = obj_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
    


class Detector(baseDet):
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.batchsize = 1
        self.weights = RosPack().get_path('deepsort') + '/scripts/yolosort/yolov4/yolov4-tiny_last.weights'
        self.config_file = RosPack().get_path('deepsort') + '/scripts/yolosort/yolov4/yolov4-tiny.cfg'
        self.data_file = RosPack().get_path('deepsort') + '/scripts/yolosort/yolov4/voc.data'
        
        network, class_names, class_colors = darknet.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            batch_size=self.batchsize
        )
        self.model = network
        self.names = class_names
        self.color = class_colors
        
    def detect(self, im):
        
        orig_h, orig_w = im.shape[:2]
        
        width = darknet.network_width(self.model)
        height = darknet.network_height(self.model)
        
        darknet_image = darknet.make_image(width, height, 3)
        
        image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.model, self.names, darknet_image, self.thresh)                 
        darknet.free_image(darknet_image)
        
        car_boxes = []
        for det in detections:
        
            pred_cls, pred_conf, (x, y, w, h) = det
            if pred_cls in ['Car', 'Misc']:
                
                pred_conf = eval(pred_conf)
                
                ori_x = round(x / width * orig_w) 
                ori_y = round(y / height * orig_h)
                ori_w = round(w / width * orig_w)
                ori_h = round(h / height * orig_h)
            
                car_boxes.append((ori_x, ori_y, ori_w, ori_h, pred_cls, pred_conf))
                           
                
        return im, car_boxes  

from yolosort.sort import *

import cv2
import numpy as np
import rospy

import time

from sort.msg import BoundingBox

mot_tracker = Sort()  

def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4
    
    for (x1, y1, x2, y2, pos_id) in bboxes:
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
        
        cv2.putText(image, 'CAR-{}'.format(pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    
        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

        ndarray_pts = np.array(list_pts, np.int32)
        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        list_pts.clear()
    return image

def update(image, bboxes):

        if len(bboxes):
            # Adapt detections to deep sort input format
            bboxes = np.array(bboxes)
            bboxes2draw = []
            
            sort_start = time.time()
            # Pass detections to sort
            # ipnut:[x1,y1,x2,y2,score]     output: [x1,y1,x2,y2,score,track_id]
            outputs = mot_tracker.update(bboxes)       
            # print(outputs)
            sort_time = time.time() - sort_start
            # print('det:{} fps, dpsort:{} fps'.format(int(1/det_time),int(1/dpsort_time)))
            
            bboxes = []
            
            for value in list(outputs):
                x1,y1,x2,y2,conf,track_id = value
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                track_id = int(track_id)
                # track_lp = ''
                bboxes2draw.append((x1, y1, x2, y2, track_id))
                
                bbox_msg = BoundingBox(conf,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        track_id,
                                        'Car')
                bboxes.append(bbox_msg)
                
            image = plot_bboxes(image, bboxes2draw)

        return image, bboxes

from yolosort.sort import *

import cv2
import numpy as np
import rospy

import time
import random
import colorsys

from sort.msg import BoundingBox

mot_tracker = Sort()  

NAMES = ['Car','Misc','Truck','rcircle','Cement_truck','Pedestrian',
        'Cyclist','gcircle','TrafficLight_Black','Bus','ycircle','rperson',
        'Tram','gperson','Trailer','TrafficLight_Red','gup','Fule_Tank',
        'TrafficLight_Green','rup','yright','rright','rleft',
        'TrafficLight_Dig','gleft','Dump_Truck','TrafficLight_Yellow','gbike']

        
def get_n_hls_colors(num):         # 生成不同框的颜色
  hls_colors = []
  i = 0
  step = 360.0 / num
  while i < 360:
    h = i
    s = 90 + random.random() * 10
    l = 50 + random.random() * 10
    _hlsc = [h / 360.0, l / 100.0, s / 100.0]
    hls_colors.append(_hlsc)
    i += step
  return hls_colors
                   
def ncolors(num):                   # 构建一个颜色list
  rgb_colors = []
  if num < 1:
    return rgb_colors
  hls_colors = get_n_hls_colors(num)
  for hlsc in hls_colors:
    _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
    r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
    rgb_colors.append([r, g, b])

  return rgb_colors
            
COLORS = ncolors(28)
pts1 = np.float32([[1,668],[845,198],[1196,157],[1913,436]]) # ori_img
pts2 = np.float32([[10,740],[10,100],[1913,100],[1913,740]]) # bev_img
M = cv2.getPerspectiveTransform(pts1, pts2) # Matrix of ori_img -> bev_img

def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4
    
    for (x1, y1, x2, y2, clss, pos_id) in bboxes:
    
        color = COLORS[clss]
        # check whether hit line                                                  每个bbox的标记点，可以用来计数
        # check_point_x = x1                                                         # check_point_x   
        # check_point_y = int(y1 + ((y2 - y1) * 0.6))              # check_point_y 
        check_point_x = int(x1 + ((x2 - x1) * 0.5))              
        check_point_y = int(y1 + ((y2 - y1) * 0.5))
        

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)          # detection box plot
        tf = max(tl - 1, 1)  # font thickness
        c2 = c1[0], c1[1] - 3
        
        cv2.putText(image, '{}-{}'.format(NAMES[clss], pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    
        cv2.circle(image, (check_point_x, check_point_y),radius=point_radius, color=(0, 0, 255))
        list_pts.clear()
    return image
    
def perspectivePoint(x, y, M):
    '''
    arg:
        point: x,y
        M : perspective matrix
    return:
        [x1,y1]
    '''
    point = (x, y)
    point = np.array(point)
    point = point.reshape(-1, 1, 2).astype(np.float32)
    new_point1 = cv2.perspectiveTransform(point, M)
    
    [xx, yy] = [round(new_point1[0][0][0]),round(new_point1[0][0][1])]
    
    return xx,yy
   
def plot_bev(img, boxes):
  
  bev_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
  for box in boxes:
    cv2.circle(bev_img, (box[0],box[1]), 10, (0, 0, 255), -1)
  
  return bev_img
   
def update(image0, bboxes):

        if len(bboxes):
            # # Adapt detections to deep sort input format
            bboxes = np.array(bboxes)
                        
            sort_start = time.time()
            # Pass detections to sort
            # ipnut:[x1,y1,x2,y2,score]     output: [x1,y1,x2,y2,score,cls,track_id]
            outputs = mot_tracker.update(bboxes)       
            # print(outputs)
            sort_time = time.time() - sort_start
            # print('det:{} fps, dpsort:{} fps'.format(int(1/det_time),int(1/dpsort_time)))
            
            bboxes = []
            bboxes2draw = []
            bboxes2bev = []
            
            for value in list(outputs):
                x1,y1,x2,y2,conf,clss,track_id = value
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                label = NAMES[int(clss)]
                track_id = int(track_id)
                # track_lp = ''
                bboxes2draw.append((x1, y1, x2, y2, int(clss), track_id))
                
                x_bottom_center, y_bottom_center = perspectivePoint((x1+x2)/2, y2, M)
                bboxes2bev.append((x_bottom_center, y_bottom_center))
                
                bbox_msg = BoundingBox(conf, x1, y1, x2, y2, track_id, label)
                bboxes.append(bbox_msg)
                
            drawed_image = plot_bboxes(image0, bboxes2draw)
            bev_image = plot_bev(image0, bboxes2bev)

        return drawed_image, bev_image, bboxes

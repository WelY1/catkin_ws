from yolosort.sort import *

import cv2
import numpy as np
import rospy

import time
import random
import colorsys

from perception_msgs.msg import DynamicObject
from perception_msgs.msg import Semantic
from perception_msgs.msg import Shape
from perception_msgs.msg import State

mot_tracker = Sort()  

NAMES = ['CAR','Misc','TRUCK','rcircle','TRUCK','PEDESTRIAN',
        'BICYCLE','gcircle','TrafficLight_Black','BUS','ycircle','rperson',
        'Tram','gperson','Trailer','TrafficLight_Red','gup','Fule_Tank',
        'TrafficLight_Green','rup','yright','rright','rleft',
        'TrafficLight_Dig','gleft','TRUCK','TrafficLight_Yellow','gbike']

TYPE = ['UNKNOWN','CAR','TRUCK','BUS','BICYCLE','MOTORBIKE','PEDESTRIAN','ANIMAL']
        
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

def plot_pv(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = round(0.002 * (1920 + 1080) / 2) + 1  # line/font thickness
    # point_radius = 4
    for (x1, y1, x2, y2, clss, pos_id, x_world, y_world) in bboxes:
    
        color = COLORS[clss]
        # check whether hit line                                   # 每个bbox的标记点，可以用来计数
        # check_point_x = int(x1 + ((x2 - x1) * 0.5))              
        # check_point_y = int(y1 + ((y2 - y1) * 0.5))
        
        # detection box plot
        cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness=1, lineType=cv2.LINE_AA)          
        cv2.putText(image, '{}*{}*({},{})'.format(NAMES[clss], pos_id, int(x_world), int(y_world)), (x1, y1-4), 0, tl / 8,
                    color, thickness=1, lineType=cv2.LINE_AA)
                    
        # cv2.circle(image, (check_point_x, check_point_y),radius=point_radius, color=(0, 0, 255))
    return image
    
def perspectivePoint(x, y, M, u0=960, v0=1080, x0=0, y0=0, dx=0.022108, dy=0.159364):
    '''
    function:
        transformer points from PV to BEV, get bev coordinate: x_bev, y_bev
        transformer points from BEV to WORLD, get bev coordinate.: x_world, y_world
    arg:
        point: x,y
        M : perspective matrix
        u0, v0: pic vector
        x0, y0: world vector
    return:
        x_bev, y_bev, x_world, y_world
    '''
    point = (x, y)
    point = np.array(point)
    point = point.reshape(-1, 1, 2).astype(np.float32)
    new_point1 = cv2.perspectiveTransform(point, M)
    
    [x_bev, y_bev] = [round(new_point1[0][0][0]),round(new_point1[0][0][1])]
    x_world = (v0 - y_bev) * dy + x0
    y_world = (u0 - x_bev) * dx + y0
    return x_bev, y_bev, x_world, y_world
   
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
            
            objects = []
            object = DynamicObject()
            bboxes2draw = []
            bboxes2bev = []
            
            for value in list(outputs):
                
                x1,y1,x2,y2,conf,clss,track_id = value
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                if NAMES[int(clss)] in TYPE:
                  label = TYPE.index(NAMES[int(clss)])
                else:
                  label = 0
                track_id = int(track_id)
                # track_lp = ''

                x_ground, y_ground, x_world, y_world = perspectivePoint((x1+x2)/2, y2, M)
                bboxes2draw.append((x1, y1, x2, y2, int(clss), track_id, x_world, y_world))
                bboxes2bev.append((x_ground, y_ground))
                
                semantic_temp = Semantic()
                semantic_temp.type = label
                semantic_temp.confidence = conf
                shape_temp = Shape()
                shape_temp.type = 0
                state_temp = State()
                state_temp.pose_covariance.pose.position.x,state_temp.pose_covariance.pose.position.y = x_world, y_world
                object.id = track_id
                object.semantic = semantic_temp
                object.shape = shape_temp
                object.state = state_temp
                objects.append(object)  

            bev_image = plot_bev(image0, bboxes2bev)   
            pv_image = plot_pv(image0, bboxes2draw)
            

        return pv_image, bev_image, objects

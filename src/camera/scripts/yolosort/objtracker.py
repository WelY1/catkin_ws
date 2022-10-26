from yolosort.sort import *

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rospkg import RosPack

import time
import random
import colorsys

from perception_msgs.msg import DynamicObject
from perception_msgs.msg import Semantic
from perception_msgs.msg import Shape
from perception_msgs.msg import State

mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

'''
NAMES = ['Car', 'Misc', 'Truck', 'rcircle', 'Cement_truck',
            'Pedestrian', 'Cyclist', 'gcircle', 'TrafficLight_Black',
            'Bus', 'ycircle', 'rperson', 'Tram', 'gperson', 'Trailer',
            'TrafficLight_Red', 'gup', 'Fule_Tank', 'TrafficLight_Green',
            'rup', 'yright', 'rright', 'rleft', 'TrafficLight_Dig',
            'gleft', 'Dump_Truck', 'TrafficLight_Yellow', 'gbike']
'''
            
TYPE = [1, 4, 2, 0, 2,
        6, 4, 0, 0,
        3, 0, 6, 3, 6, 2,
        0, 0, 2, 0,
        0, 0, 0, 0, 0,
        0, 2, 0, 4]
            
TYPE_Vis = ['UNKNOWN', 'CAR', 'TRUCK', 'BUS',
         'BICYCLE', 'MOTORBIKE', 'PEDESTRIAN', 'ANIMAL']
         
         
COLORS = [[0,0,0],         # 黑色
          [220,20,60],     # 猩红
          [65,105,225],    # 皇家蓝
          [170,224,230],   # 火药蓝
          [34,139,34],     # 森林绿
          [34,139,34],     
          [255,160,122],   # 浅鲜色
          [0,0,0]
         ]


pts1 = np.float32([[775,631],[2101,612],[1349,320],[935,328]]) # ori_img
pts2 = np.float32([[0,850],[1919,850],[1919,0],[0,0]]) # bev_img
M = cv2.getPerspectiveTransform(pts1, pts2)  # Matrix of ori_img -> bev_img

font_path = RosPack().get_path('camera') + '/scripts/yolosort/simhei.ttf'
font = ImageFont.truetype(font_path, 22, encoding="utf-8") 

'''
本身用PILLOW相比于OPENCV速度很慢，
需要
# must remove existed pillow first.
$ pip uninstall pillow
# install SSE4 version
$ pip install pillow-simd
# install AVX2 version
$ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

'''
def plot_pv(image, bboxes, line_thickness=3):
    # Plots one bounding box on image img
    # tl = line_thickness  # line/font thickness
    
    start = time.time()
    pilimg = Image.fromarray(image)
    draw = ImageDraw.Draw(pilimg) 

    print((time.time()-start)*1000)
    
    start = time.time()
    for (x1, y1, x2, y2, label, pos_id, lp, x_world, y_world) in bboxes:
        
        color1 = tuple(COLORS[label])
       
        # plot bbox
        #  c1, c2 = (x1, y1), (x2, y2)
        # draw.rectangle((c1,c2),fill=None,outline=color1,width=line_thickness)         # need pillow>=5.3.0
        draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)],  width=line_thickness, fill=color1)
        # plot x, y        
        txt2 = '({:.2f},{:.2f})'.format(x_world, y_world)
        t_size2 = font.getsize(txt2)
        c1 = x1, y1 - t_size2[1]
        c2 = x1 + t_size2[0] , y1 
        draw.rectangle((c1,c2),fill=color1,outline=None)
        draw.text((c1[0],c1[1]-1),txt2,(255,255,255),font=font)
        # plot label-id-lp
        if lp == 'UnKnown':
            txt1 = '{}-{}'.format(TYPE_Vis[label], pos_id)
        else:
            txt1 = '{}-{}-{}'.format(TYPE_Vis[label], pos_id, lp)
        t_size1 = font.getsize(txt1)
        c2 = x1 + t_size1[0], c1[1]-1
        c1 = x1, c1[1]- t_size1[1]-2

        draw.rectangle((c1,c2),fill=color1,outline=None)
        draw.text((c1[0],c1[1]-1),txt1,(255,255,255),font=font)
        
    print((time.time()-start)*1000)
    start = time.time()
    npimg = np.array(pilimg)
    print((time.time()-start)*1000)             
        
    return npimg

'''
# opencv画图速度快,但是不能显示中文
def plot_pv(image, bboxes, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness  # line/font thickness
    
    for (x1, y1, x2, y2, label, pos_id, lp, x_world, y_world) in bboxes:
        
        color1 = COLORS[label]
        c1, c2 = (x1, y1), (x2, y2)
        # detection box plot
        cv2.rectangle(image, c1, c2, color1,
                      thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)
        
        txt2 = '({:.2f},{:.2f})'.format(x_world, y_world)
        t_size2 = cv2.getTextSize(txt2, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = x1, y1 - t_size2[1]
        c2 = x1 + t_size2[0] , y1 
        cv2.rectangle(image, c1, c2, color1, cv2.FILLED) 
        cv2.putText(image, txt2, (c1[0],c2[1]-1), 0, tl/3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)
        
        if lp == 'UnKnown':
            txt1 = '{}-{}'.format(TYPE_Vis[label], pos_id)
        else:
            txt1 = '{}-{}-{}'.format(TYPE_Vis[label], pos_id, lp)
        t_size1 = cv2.getTextSize(txt1, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size1[0], c1[1]-1
        c1 = x1, c1[1]- t_size1[1]-2
        
        cv2.rectangle(image, c1, c2, color1, cv2.FILLED) 
        cv2.putText(image, txt1, (c1[0],c2[1]), 0, tl/3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)

    return image
'''

def perspectivePoint(x, y, M, u0=495, v0=917, x0=20.4649, y0=-1.27286, dx=0.010329, dy=0.065753):
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

    [x_bev, y_bev] = [round(new_point1[0][0][0]), round(new_point1[0][0][1])]
    x_world = (v0 - y_bev) * dy + x0
    y_world = (u0 - x_bev) * dx + y0
    return x_bev, y_bev, x_world, y_world


def plot_bev(img, boxes):

    bev_img = cv2.warpPerspective(
        img, M, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    for box in boxes:
        cv2.circle(bev_img, (box[0], box[1]), 10, (0, 0, 255), -1)
    return bev_img


def update(image0, bboxes):

    if len(bboxes):
        # # Adapt detections to deep sort input format
        bboxes = np.array(bboxes)

        start2 = time.time()
        # Pass detections to sort
        # ipnut:[x1,y1,x2,y2,score]     output: [x1,y1,x2,y2,score,cls,track_id]
        outputs = mot_tracker.update(image0, bboxes)
        # print(outputs)
        time2 = time.time() - start2
        # print('det:{} fps, dpsort:{} fps'.format(int(1/det_time),int(1/dpsort_time)))

        objects = []
        bboxes2draw = []
        bboxes2bev = []

        for value in list(outputs):

            x1, y1, x2, y2, conf, clss, track_lp, track_id = value
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            track_id = int(track_id)
            
            label = int(TYPE[int(clss)])
            
            # start3 = time.time()
            x_ground, y_ground, x_world, y_world = perspectivePoint((x1+x2)/2, y2, M)
            # time3 = time.time() - start3
            
            bboxes2draw.append((x1, y1, x2, y2, label, track_id, track_lp, x_world, y_world))
            bboxes2bev.append((x_ground, y_ground))
            
            semantic_temp = Semantic()
            semantic_temp.type = label
            semantic_temp.confidence = conf
            semantic_temp.lp = track_lp
            shape_temp = Shape()
            shape_temp.type = 0
            #add 20221009
            if  semantic_temp.type == 1: # car
                shape_temp.dimensions.x = 4.0
                shape_temp.dimensions.y = 1.6
                shape_temp.dimensions.z = 1.5
                #shape_temp.type = 0
            elif semantic_temp.type ==2 or semantic_temp.type ==3: # truck bus
                shape_temp.dimensions.x = 6.0
                shape_temp.dimensions.y = 2.0
                shape_temp.dimensions.z = 3.0
            elif semantic_temp.type ==4 or semantic_temp.type ==5: # bicycle motorbike
                shape_temp.dimensions.x = 1.6
                shape_temp.dimensions.y = 0.7
                shape_temp.dimensions.z = 1.5
            else: #ped ...
                shape_temp.dimensions.x = 0.5
                shape_temp.dimensions.y = 0.5
                shape_temp.dimensions.z = 1.7
                shape_temp.type = 1         # ped is cylinder
            # add end 
            
            state_temp = State()
            state_temp.pose_covariance.pose.position.x, state_temp.pose_covariance.pose.position.y = x_world, y_world
            object = DynamicObject()
            object.id = track_id
            object.semantic = semantic_temp
            object.shape = shape_temp
            object.state = state_temp
            objects.append(object)

        # bev_image = plot_bev(image0, bboxes2bev)
        start4 = time.time()
        pv_image = plot_pv(image0, bboxes2draw)
        time4 = time.time() - start4

        return pv_image, objects, time2, time4


if __name__ == '__main__':
    # point = (1133,717)           # (1055,717) and (1210,717)
    # point = np.array(point)
    # point = point.reshape(-1, 1, 2).astype(np.float32)
    # new_point1 = cv2.perspectiveTransform(point, M)   # bev:(495,917)   world:(20.4649,-1.27286)
    # print(new_point1)
    points=[(8,1044),(86,644),(267,642),(170,519),(622,517),(511,429),(613,427),(637,341),
            (873,250),(897,325),(994,335),(837,514),(546,1070),(1916,749),(1576,497),(1491,475),
            (1525,388),(1356,390),(1318,366),(1125,269),(1001,236),(1042,1076),(1541,1076),(1906,962),
            (1518,629),(1266,632),(1021,633),(781,634),(806,604),(835,542),(1526,529),(1620,592),
            (1119,325),(1026,226)]
    i = 1
    for point in points:
        x_ground, y_ground, x_world, y_world = perspectivePoint(point[0],point[1], M)
        print('{}  {}  ({:.2f},{:.2f})'.format(i,point,x_world, y_world))
        i += 1
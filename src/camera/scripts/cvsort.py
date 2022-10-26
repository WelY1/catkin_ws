#! /usr/bin/python

import rospy
import cv2
import time

from yolosort.objdetector import TrtYOLO as Detector

from cv_bridge import CvBridge
# import ros_numpy
import numpy as np

from sensor_msgs.msg import Image
from camera.msg import Bbox
from camera.msg import Bboxes

'''
OBJ_LIST = ['Car', 'Misc', 'Truck', 'rcircle', 'Cement_truck', 
            'Pedestrian', 'Cyclist', 'gcircle', 'TrafficLight_Black', 'Bus', 
            'ycircle', 'rperson', 'Tram', 'gperson', 'Trailer', 'TrafficLight_Red',
            'gup', 'Fule_Tank', 'TrafficLight_Green', 'rup', 'yright', 'rright', 'rleft',
            'TrafficLight_Dig', 'gleft', 'Dump_Truck', 'TrafficLight_Yellow', 'gbike']
'''

conf_th = 0.3
conf_th_truck = 0.7
conf_matrix = [conf_th, conf_th, conf_th_truck, conf_th, conf_th_truck, conf_th,
               conf_th, conf_th, conf_th, conf_th_truck, conf_th, conf_th,
               conf_th_truck, conf_th, conf_th_truck, conf_th, conf_th, conf_th_truck,
               conf_th, conf_th, conf_th, conf_th, conf_th,
               conf_th, conf_th, conf_th_truck, conf_th, conf_th]

def callback_image(msg):
    global avg1
    global avg2
    global avg3
    global avg4
    global avg5
    global avg6
    global avg7
    
    start1 = time.time()
    # frame_img = ros_numpy.numpify(msg)   # [1080,1920,4]
    # frame_img = np.ascontiguousarray(frame_img[:,:,:3])  # [1080,1920,4] -> [1080,1920,3]
    frame_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    time1 = time.time()-start1
    
    
    # detect and sort
    start5 = time.time()
    det_bboxes,time2,time3,time4 = det.detect(frame_img)
    time5 = time.time() - start5
    
    start6 = time.time()
    boxes_temp = []
    for det_box in det_bboxes:
        bbox = Bbox()
        bbox.x1,bbox.y1,bbox.x2,bbox.y2,bbox.conf,bbox.clss = det_box
        if bbox.conf >= conf_matrix[bbox.clss]:
            boxes_temp.append(bbox)
        
     
    # publish
    temp_msg.header = msg.header  
    temp_msg.bboxes = boxes_temp
    temp_msg.image = msg
    boxes_pub.publish(temp_msg)
    time6 = time.time() - start6
 
    time7 = time.time() - start1
    avg1 = time1 if avg1==0 else avg1 * 0.95 + time1 * 0.05   # cv_bridge
    avg2 = time2 if avg2==0 else avg2 * 0.95 + time2 * 0.05   # yolov4 preprocess
    avg3 = time3 if avg3==0 else avg3 * 0.95 + time3 * 0.05   # yolov4 engine infer
    avg4 = time4 if avg4==0 else avg4 * 0.95 + time4 * 0.05   # yolov4 postprocess
    avg5 = time5 if avg5==0 else avg5 * 0.95 + time5 * 0.05   # yolov4 total
    avg6 = time6 if avg6==0 else avg6 * 0.95 + time6 * 0.05   # msg process and publish
    avg7 = time7 if avg7==0 else avg7 * 0.95 + time7 * 0.05   # total time
    # print('*****{} fps*****'.format(int(1/(time.time()-start1))), end='\r')
    print('----------------------------------------------------')
    print('cv_bridge / ros_numpy   : {:.2f} ms'.format(avg1 * 1000))
    print('yolov4 preprocess       : {:.2f} ms'.format(avg2 * 1000))
    print('yolov4 engine infer     : {:.2f} ms'.format(avg3 * 1000))
    print('yolov4 postprocess      : {:.2f} ms'.format(avg4 * 1000))
    print('yolov4 total            : {:.2f} ms'.format(avg5 * 1000))
    print('msg process and publish : {:.2f} ms'.format(avg6 * 1000))
    print('total time              : {:.2f} ms'.format(avg7 * 1000))
    
    rate.sleep()

def main():
        
    rospy.init_node('cvsort', anonymous=True)
    global rate
    rate = rospy.Rate(40)
     
    global det
    global temp_msg
    global boxes_pub
    global img_pub
    global bridge
    bridge = CvBridge()
    '''调试用'''
    global avg1
    avg1 = 0
    global avg2
    avg2 = 0
    global avg3
    avg3 = 0
    global avg4
    avg4 = 0
    global avg5
    avg5 = 0
    global avg6
    avg6 = 0
    global avg7
    avg7 = 0
    
    det = Detector(conf_th=conf_th, nms_th=0.5)        
    temp_msg = Bboxes()
    # subscribe image
    image_sub = rospy.Subscriber('/neuvition_image', Image, callback_image, queue_size=1, buff_size=1920*1080*3)
    boxes_pub = rospy.Publisher('/temp_msg', Bboxes, queue_size=1)

    rospy.spin()
    

if __name__ == '__main__':
    
    main()
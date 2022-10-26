#! /usr/bin/python

'''
origin model
'''
import rospy
import cv2
import time

import yolosort.objtracker as Mot

from cv_bridge import CvBridge
# import ros_numpy
import numpy as np

from sensor_msgs.msg import Image
from perception_msgs.msg import DynamicObjectArray
from camera.msg import Bboxes

def callback_det(temp_msg):
    global avg1
    global avg2
    global avg3
    global avg4
    global avg5
    global avg6
    
    start1 = time.time()
    frame_img = bridge.imgmsg_to_cv2(temp_msg.image, desired_encoding="bgr8")
    # frame_img = ros_numpy.numpify(temp_msg.image)   # [1080,1920,4]
    # frame_img = np.ascontiguousarray(frame_img[:,:,:3])  # [1080,1920,4] -> [1080,1920,3]
    
    boxes = []
    for bbox in temp_msg.bboxes:
        box = (int(bbox.x1),int(bbox.y1),int(bbox.x2),int(bbox.y2),bbox.conf,int(bbox.clss))
        boxes.append(box)
    time1 = time.time()-start1
    
    # sort
    drawed_img, object_temp,time2,time4 = Mot.update(frame_img, boxes)
    
    start5 = time.time()
    # publish
    cam_msg.header = temp_msg.header  
    cam_msg.objects = object_temp
    boxes_pub.publish(cam_msg)
    
    img_msg = bridge.cv2_to_imgmsg(drawed_img, encoding="bgr8")
    # img_msg = ros_numpy.msgify(Image, drawed_img, encoding='bgr8')
    img_msg.header = temp_msg.header
    image_pub.publish(img_msg)
    time5 = time.time() - start5
    time6 = time.time() - start1
    
    avg1 = time1 if avg1==0 else avg1 * 0.95 + time1 * 0.05   # temp_msg process
    avg2 = time2 if avg2==0 else avg2 * 0.95 + time2 * 0.05   # pure sort
    # avg3 = time3 if avg3==0 else avg3 * 0.95 + time3 * 0.05   # perspectivePoint
    avg4 = time4 if avg4==0 else avg4 * 0.95 + time4 * 0.05   # plot pv
    avg5 = time5 if avg5==0 else avg5 * 0.95 + time5 * 0.05   # msg publish
    avg6 = time6 if avg6==0 else avg6 * 0.95 + time6 * 0.05   # total time
    # avg11 = time11 if avg11==0 else avg11 * 0.95 + time11 * 0.05   # yolov5 preprocess
    # avg12 = time12 if avg12==0 else avg12 * 0.95 + time12 * 0.05   # yolov5 engine infer
    # avg13 = time13 if avg13==0 else avg13 * 0.95 + time13 * 0.05   # yolov5 postprocess
    # avg14 = time14 if avg14==0 else avg14 * 0.95 + time14 * 0.05   # yolov5 total
    # avg21 = time21 if avg21==0 else avg21 * 0.95 + time21 * 0.05   # ocr preprocess
    # avg22 = time22 if avg22==0 else avg22 * 0.95 + time22 * 0.05   # ocr net infer
    # avg23 = time23 if avg23==0 else avg23 * 0.95 + time23 * 0.05   # ocr postprecess
    # avg24 = time24 if avg24==0 else avg24 * 0.95 + time24 * 0.05   # ocr total
    
    
    print('----------------------------------------------------')
    print('temp_msg process        : {:.2f} ms'.format(avg1 * 1000))
    print('pure sort               : {:.2f} ms'.format(time2 * 1000))
    print('perspectivePoint        : {:.2f} ms'.format(avg3 * 1000))
    print('plot pv                 : {:.2f} ms'.format(avg4 * 1000))
    print('msg publish             : {:.2f} ms'.format(avg5 * 1000))
    print('total time              : {:.2f} ms'.format(time6 * 1000))
    print('*****{} fps*****'.format(int(1/time6)))
    # print('yolov5 preprocess       : {:.2f} ms'.format(avg11 * 1000))
    # print('yolov5 engine infer     : {:.2f} ms'.format(avg12 * 1000))
    # print('yolov5 postprocess      : {:.2f} ms'.format(avg13 * 1000))
    # print('yolov5 total            : {:.2f} ms'.format(avg14 * 1000))
    # print('ocr preprocess          : {:.2f} ms'.format(avg21 * 1000))
    # print('ocr net infer           : {:.2f} ms'.format(avg22 * 1000))
    # print('ocr postprecess         : {:.2f} ms'.format(avg23 * 1000))
    # print('ocr total               : {:.2f} ms'.format(avg24 * 1000))
    
    rate.sleep()
    

def main():
        
    rospy.init_node('lpsort', anonymous=True)
    global rate
    rate = rospy.Rate(40)
     
    global cam_msg
    global boxes_pub
    global image_pub 
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
    
    cam_msg = DynamicObjectArray()
    boxes_pub = rospy.Publisher('/cam_msg', DynamicObjectArray, queue_size=1)
    image_pub = rospy.Publisher('/cam_img', Image, queue_size=1)
    # subscribe image
    image_sub = rospy.Subscriber('/temp_msg', Bboxes, callback_det, queue_size=1)
    rospy.spin()
    

if __name__ == '__main__':
    
    main()
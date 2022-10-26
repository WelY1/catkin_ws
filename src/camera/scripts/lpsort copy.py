#! /usr/bin/python

'''
origin model
'''
import rospy
import cv2
import time

import yolosort.objtracker as Mot

import ros_numpy
import numpy as np

from sensor_msgs.msg import Image
from perception_msgs.msg import DynamicObjectArray
from camera.msg import Bboxes

def callback_det(temp_msg):

    frame_img = ros_numpy.numpify(temp_msg.image)   # [1080,1920,4]
    frame_img = np.ascontiguousarray(frame_img[:,:,:3])  # [1080,1920,4] -> [1080,1920,3]
    
    # detect and sort
    start = time.time()
    boxes = []
    for bbox in temp_msg.bboxes:
        box = (bbox.x1,bbox.y1,bbox.x2,bbox.y2,bbox.conf,bbox.clss)
        boxes.append(box)
    drawed_img, bev_image, object_temp = Mot.update(frame_img, boxes)
    
    # publish
    cam_msg.header = temp_msg.header  
    cam_msg.objects = object_temp
    boxes_pub.publish(cam_msg)

    img_msg = ros_numpy.msgify(Image, drawed_img, encoding='bgr8')
    img_msg.header = temp_msg.header
    image_pub.publish(img_msg)
    
    # 可视化
    cv2.namedWindow('bev_img',0)
    cv2.resizeWindow('bev_img',900,500)
    cv2.imshow('bev_img',bev_image)  
    cv2.namedWindow('ori_img',0)
    cv2.resizeWindow('ori_img',900,500)
    cv2.imshow('ori_img',drawed_img)  
    cv2.waitKey(1)
    
    print('*****{:1f} fps*****'.format(1/(time.time()-start)), end='\r')
    rate.sleep()
    

def main():
        
    rospy.init_node('lpsort', anonymous=True)
    global rate
    rate = rospy.Rate(20)
     
    global cam_msg
    global boxes_pub
    global image_pub    
    
    cam_msg = DynamicObjectArray()
    boxes_pub = rospy.Publisher('/cam_msg', DynamicObjectArray, queue_size=1)
    image_pub = rospy.Publisher('/cam_img', Image, queue_size=1)
    # subscribe image
    image_sub = rospy.Subscriber('/temp_msg', Bboxes, callback_det, queue_size=1)
    rospy.spin()
    

if __name__ == '__main__':
    
    main()
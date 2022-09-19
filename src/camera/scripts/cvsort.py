#! /usr/bin/python

import rospy
import cv2
import time

from yolosort.objdetector import TrtYOLO as Detector
import yolosort.objtracker as Mot

# from cv_bridge import CvBridge
import ros_numpy
import numpy as np

from sensor_msgs.msg import Image
from perception_msgs.msg import DynamicObjectArray

def callback_image(msg):

    # frame_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    frame_img = ros_numpy.numpify(msg)   # [1080,1920,4]
    frame_img=np.ascontiguousarray(frame_img[:,:,:3])  # [1080,1920,4] -> [1080,1920,3]
    
    # detect and sort
    start = time.time()
    boxes = det.detect(frame_img)
    drawed_img, bev_image, object_temp = Mot.update(frame_img, boxes)
    
    # publish
    cam_msg.header = msg.header  
    cam_msg.objects = object_temp
    boxes_pub.publish(cam_msg)
    # img_msg = bridge.cv2_to_imgmsg(drawed_img, encoding="bgr8")
    # print(drawed_img.shape)
    img_msg = ros_numpy.msgify(Image, drawed_img, encoding='bgr8')
    img_msg.header = msg.header
    image_pub.publish(img_msg)
    
    # # 可视化
    # cv2.namedWindow('bev_img',0)
    # cv2.resizeWindow('bev_img',900,500)
    # cv2.imshow('bev_img',bev_image)  
    # cv2.namedWindow('ori_img',0)
    # cv2.resizeWindow('ori_img',900,500)
    # cv2.imshow('ori_img',drawed_img)  
    # cv2.waitKey(1)
    
    print('*****{:1f} fps*****'.format(1/(time.time()-start)), end='\r')
    rate.sleep()
    

def main():
        
    rospy.init_node('cvsort', anonymous=True)
    global rate
    rate = rospy.Rate(20)
     
    global det
    global cam_msg
    global boxes_pub
    global image_pub
    # global bridge
    # bridge = CvBridge()
    det = Detector()        
    
    cam_msg = DynamicObjectArray()
    boxes_pub = rospy.Publisher('/cam_msg', DynamicObjectArray, queue_size=1)
    image_pub = rospy.Publisher('/cam_img', Image, queue_size=1)
    # subscribe image
    image_sub = rospy.Subscriber('/neuvition_image', Image, callback_image, queue_size=1, buff_size=1960*1080*3)
    rospy.spin()
    

if __name__ == '__main__':
    
    main()
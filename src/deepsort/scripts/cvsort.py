#! /usr/bin/python

import rospy
import imutils
import cv2

from yolosort.objdetector import Detector
import yolosort.objtracker as mot

from cv_bridge import CvBridge

from deepsort.msg import Bboxes
from sensor_msgs.msg import Image


def callback_image(msg):
    global bridge
    global det
    # global mot
    global boxes_msg
    global rate
    frame_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    boxes = det.detect(frame_img)
    drawed_img, boxes_msg = mot.update(frame_img, boxes)
    
    drawed_img = imutils.resize(drawed_img, height=500)
    cv2.imshow('demo',drawed_img)
    t = int(1000/20)
    cv2.waitKey(t)
    
    box_pub.publish(boxes_msg)
    rate.sleep()
    
def main():
        
    rospy.init_node('cvsort', anonymous=True)
    global rate
    rate = rospy.Rate(20)
     
    global bridge
    global det
    global boxes_msg
    global box_pub
    
    bridge = CvBridge()
    det = Detector()        # 实例化检测器
    
    box_pub = rospy.Publisher('boundingboxes', Bboxes, queue_size=10)
    # subscribe image
    image_sub = rospy.Subscriber('frame_image', Image, callback_image, queue_size=1, buff_size=1960*1080*3)
    rospy.spin()
    
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()
#! /usr/bin/python

import rospy
import imutils
import cv2
import time

from yolosort.objdetector import TrtYOLO as Detector
import yolosort.objtracker as mot

from cv_bridge import CvBridge

from sort.msg import BoundingBoxes
from sensor_msgs.msg import Image


def callback_image(msg):
    global bridge
    global det
    # global mot
    global boxes_msg
    global rate
    frame_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # cv2.imshow('ori_img', frame_img)
    # h,w = frame_img.shape[:2]
    # print(h,w)
    # global videoWriter
    # videoWriter.write(frame_img)
    
    start = time.time()
    boxes = det.detect(frame_img)
    drawed_img, bboxes = mot.update(frame_img, boxes)
    
    boxes_msg.header.stamp = rospy.Time.now()
    boxes_msg.image_header.stamp = rospy.Time.now()
    boxes_msg.bounding_boxes = bboxes
    
    print('*****{} fps*****'.format(1/(time.time()-start)))
    
    drawed_img = imutils.resize(drawed_img, height=500)
    cv2.imshow('demo',drawed_img)
    t = int(1000/20)
    cv2.waitKey(t)

    boxes_pub.publish(boxes_msg)
    rate.sleep()
    
def main():
        
    rospy.init_node('cvsort', anonymous=True)
    global rate
    rate = rospy.Rate(20)
     
    global bridge
    global det
    global boxes_msg
    global boxes_pub
    
    bridge = CvBridge()
    det = Detector()        
    
    # global videoWriter
    # f = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')#VideoWriter_fourcc为视频编解码器
    # videoWriter = cv2.VideoWriter('/home/zxc/catkin_ws/src/sort/result/2.mp4', f, 20, (1920, 1080))

    boxes_msg = BoundingBoxes()
    boxes_pub = rospy.Publisher('boundingboxes', BoundingBoxes, queue_size=10)
    # subscribe image
    image_sub = rospy.Subscriber('/neuvition_image', Image, callback_image, queue_size=1, buff_size=1960*1080*3)
    rospy.spin()
    

if __name__ == '__main__':
    
    main()
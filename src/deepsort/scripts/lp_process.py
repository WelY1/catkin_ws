#! /usr/bin/python
#!coding=utf-8
  
import rospy
import torch

from cv_bridge import CvBridge

from image_trans.msg import Lp
from image_trans.msg import Result

from yolov5.lpdetector import Detector
from lp.recognition import Recognition

def resultPub(id, lp, conf):
    result_pub = rospy.Publisher("resultpub", Result, queue_size=10)
    rate = rospy.Rate(20)
    result = Result()
    result.id = id
    result.lp = lp
    result.conf = conf

    result_pub.publish(result)
    rate.sleep()
    rospy.loginfo('%d--%s--%f  已发送', result.id, result.lp, result.conf)

def callback(msg):     # 无限循环

    global bridge
    global det
    global ocr
        
    car_img = bridge.imgmsg_to_cv2(msg.lp, "bgr8")
    id = msg.id
    
    # cv2.imshow("img" , car_img)
    # cv2.waitKey(1)
    
    # print('..')
    with torch.no_grad():
        lpbox = det.detect(car_img)
        
        if len(lpbox):
            x1, y1, x2, y2 = lpbox[0], lpbox[1], lpbox[2], lpbox[3]
            im_crops = []
            im_crops.append(car_img[y1:y2,x1:x2])
            lp, conf = ocr(im_crops)
            rospy.loginfo('车辆id: %d, 车牌号: %s, 置信度: %f', id, lp, conf)
            resultPub(id, lp, conf)
            
    # rospy.loginfo(id)

 
def displayWebcam():                            # 只执行一次
    rospy.init_node('lpprocess', anonymous=True)
 
    global bridge   
    global det
    global ocr
    det = Detector()    # imput: images  -> outputs: images
    ocr = Recognition()

    bridge = CvBridge()

    rospy.Subscriber('lp', Lp, callback, queue_size=20, buff_size=1920*1080*3)
    rospy.spin()
 
if __name__ == '__main__':

    displayWebcam()          # 只执行一次
 

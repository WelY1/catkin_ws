#! /usr/bin/python

import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from rospkg import RosPack
VIDEO_PATH = RosPack().get_path('deepsort') + '/scripts/yolosort/video/test_traffic.mp4'

def main():
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    # fps = int(cap.get(5))
    # t = int(1000/fps)
    
    rospy.init_node('pub', anonymous=True)
    
    img_pub = rospy.Publisher('frame_image', Image, queue_size=1)
    rate = rospy.Rate(20)
    bridge = CvBridge()
    
    while True:
        _, im = cap.read()
        if im is None:
            break
        img = bridge.cv2_to_imgmsg(im, encoding="bgr8")
        img_pub.publish(img)
        rate.sleep()
        
    cap.release()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

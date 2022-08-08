#! /usr/bin/python

import rospy
import imutils
import cv2

rospy.init_node('yolosort', anonymous=True)
from yolosort.objdetector import Detector

from rospkg import RosPack
from deepsort.msg import Bboxes

VIDEO_PATH = RosPack().get_path('deepsort') + '/scripts/yolosort/video/test_traffic.mp4'
RESULT_PATH = RosPack().get_path('deepsort') + '/scripts/result.mp4'

def main():
       
    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()                                                           # 实例化检测器
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    size = None
    videoWriter = None
    
    pub = rospy.Publisher('boundingboxes', Bboxes, queue_size=10)
    
    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im, func_status)
        frame_img = result['frame']
        frame_img = imutils.resize(frame_img, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (frame_img.shape[1], frame_img.shape[0]))

        videoWriter.write(frame_img)
        cv2.imshow(name, frame_img)
        cv2.waitKey(t)
        
        # 发布当前帧的bboxes结果
        bboxes_msg  = result['obj_bboxes']
        pub.publish(bboxes_msg)
        

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()
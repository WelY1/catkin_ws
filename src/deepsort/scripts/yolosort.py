#! /usr/bin/python

import rospy
import imutils
import cv2

rospy.init_node('yolosort', anonymous=True)
from yolosort.objdetector import Detector

from rospkg import RosPack

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

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()
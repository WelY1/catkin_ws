import rospy
import cv2
import time

from yolosort.objdetector import TrtYOLO as Detector
import yolosort.objtracker as Mot

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

VIDEO_PATH = '/home/zxc/catkin_ws/src/video/1.mp4'

def main():

    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()      

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    k = 0
    while True:

        # try:
        _, im = cap.read()
        k += 1
        if im is None:
            print(k)
            break
        
        start = time.time()
        boxes = det.detect(im)
        print(boxes)
        drawed_img, bev_image, bboxes = Mot.update(im, boxes)
        
        
        # 可视化
        cv2.namedWindow('bev_img',0)
        cv2.resizeWindow('bev_img',900,500)
        cv2.imshow('bev_img',bev_image)  
        cv2.namedWindow('ori_img',0)
        cv2.resizeWindow('ori_img',900,500)
        cv2.imshow('ori_img',drawed_img)  
        cv2.waitKey(1)
        
        print('*****{:1f} fps*****'.format(1/(time.time()-start)), end='\r')


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()
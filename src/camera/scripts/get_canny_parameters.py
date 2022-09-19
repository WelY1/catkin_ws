'''
To find the parameters of Canny detection
'''

import cv2
import numpy as np

cv2.namedWindow('edge_detection',0)
cv2.resizeWindow('edge_detection',500,800)
cv2.createTrackbar('minThreshold','edge_detection',50,1000,lambda x: x)
cv2.createTrackbar('maxThreshold','edge_detection',100,1000,lambda x: x)
img_path = '/home/zxc/yutong/Yolov4_DeepSocial/ori_img.jpg'
img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

mask=np.zeros_like(img)   #变换为numpy格式的图片
# mask=cv2.fillPoly(mask,np.array([[[850,600],[797,773],[1165,773],[1093,600]]]),color=255)   #对感兴趣区域制作掩膜   find_scale_x
# mask=cv2.fillPoly(mask,np.array([[[1186,393],[1197,393],[1213,435],[1205,435]]]),color=255)   #对感兴趣区域制作掩膜    find_scale_y  
mask=cv2.fillPoly(mask,np.array([[[777,483],[820,479],[849,427],[813,435]]]),color=255)   #对感兴趣区域制作掩膜    find_scale_y

#在此做出说明，实际上，相机固定于一个位置，所以对于感兴趣的区域的位置也相对固定，这个视相机位置而定。
cv2.namedWindow('mask',0)
cv2.resizeWindow('mask',800,1200)
cv2.imshow('mask',mask)
# cv2.waitKey(0)
masked_edge_img=cv2.bitwise_and(img,mask)   #与运算
while True:
    minThreshold=cv2.getTrackbarPos('minThreshold','edge_detection')
    maxThreshold=cv2.getTrackbarPos('maxThreshold','edge_detection')
    edges=cv2.Canny(masked_edge_img,minThreshold,maxThreshold)
    cv2.imshow('edge_detection',edges)
    cv2.waitKey(10)

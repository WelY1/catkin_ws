'''
Only run once to get scale from real world to pixel
detection lane in ROI and transform to bev image to get pixel distance
the width of lane with 60km/h  is 3.5m
faster than 60km/h is 3.75m
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawlines(img0, lines):
    k = 0
    for line in lines:
        
        x_1,y_1,x_2,y_2=line[0]
        cv2.line(img0, (x_1,y_1),(x_2,y_2), ((100+50*k)%255,(0+120*k)%255,(0-60*k)%255),1)
        k += 1

def draw(img0, line, color=(255,0,0), tl=3):
    x_1,y_1,x_2,y_2=line[0][0],line[0][1],line[1][0],line[1][1]
    cv2.line(img0, (x_1,y_1),(x_2,y_2), color, tl)


def calculate_slope(line):
    '''计算线段line的斜率
    ：param Line：np.array([[x_1,y_1,x_2,y_2]])
    :return:
    '''
    x_1,y_1,x_2,y_2=line[0]
    return (y_2-y_1)/(x_2-x_1)

def reject_abnormal_lines(lines,threshold):
    '''剔出斜率不一致的线段'''
    slopes=[calculate_slope(line) for line in lines]
    while len(lines)>0:
        mean=np.mean(slopes)
        diff=[abs(s-mean) for s in slopes]
        idx=np.argmax(diff)
        if diff[idx]>threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines
# print(len(right_lines),len(left_lines))

def least_squares_fit(lines):

    x_coords=np.ravel([[line[0][0],line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])   #取出所有标点
    poly=np.polyfit(x_coords,y_coords,deg=1)                             #进行直线拟合，得到多项式系数
    point_min=(np.min(x_coords),np.polyval(poly,np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))     #根据多项式系数，计算两个直线上的点
    
    return np.array([point_min,point_max],dtype=np.int64)


def perspectivePoint(line, M):
    '''
    arg:
        line: [[x1,y1],[x2,[y2]]]
        M : perspective matrix
    return:
        [x1,y1],[x2,y2]
    '''
    x1,y1,x2,y2=line[0][0],line[0][1],line[1][0],line[1][1]
    point1 = (x1,y1)
    point1 = np.array(point1)
    point1 = point1.reshape(-1, 1, 2).astype(np.float32)
    new_point1 = cv2.perspectiveTransform(point1, M)
    
    point2 = (x2,y2)
    point2 = np.array(point2)
    point2 = point2.reshape(-1, 1, 2).astype(np.float32)
    new_point2 = cv2.perspectiveTransform(point2, M)
    
    p1 = [round(new_point1[0][0][0]),round(new_point1[0][0][1])]
    p2 = [round(new_point2[0][0][0]),round(new_point2[0][0][1])]
    
    return p1, p2 
    

def get_distance_point2line(point, line):
    """
    Args:
        point: [x0, y0]
        line: [x1, y1, x2, y2]
    """
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def find_scale_x():
    '''1.canny边缘检测'''
    img_path = '/home/zxc/catkin_ws/src/camera/scripts/yutong.jpg'
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)     #以灰度图形式读取图片，为canny边缘检测做准备
    img0=cv2.imread(img_path,cv2.IMREAD_COLOR)
    
    edge_img=cv2.Canny(img,82,126)     #设定阈值，低于阈值被忽略，高于阈值被显示，
                                           # 阈值的设定与图片的色彩有关，需要手动调整到合适的值（使车道线清晰显示出来）

    '''2.roi_mask(提取感兴趣的区域)'''
    mask=np.zeros_like(edge_img)   #变换为numpy格式的图片
    mask=cv2.fillPoly(mask,np.array([[[1000,813],[1425,813],[1452,847],[1004,853]]]),color=255)   #对感兴趣区域制作掩膜
    #在此做出说明，实际上，车载相机固定于一个位置，所以对于感兴趣的区域的位置也相对固定，这个视相机位置而定。
    # cv2.namedWindow('mask',0)
    # cv2.resizeWindow('mask',800,1200)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    masked_edge_img=cv2.bitwise_and(edge_img,mask)   #与运算
    # cv2.imshow('mask',masked_edge_img)
    # cv2.waitKey(0)
    
    '''3.霍夫变换，找出直线'''
    
    lines=cv2.HoughLinesP(masked_edge_img,1,np.pi/180,15,minLineLength=25,maxLineGap=10)    #获取所有线段
    # print(lines)
    # right_lines=[line for line in lines if calculate_slope(line)>0]
    # print(right_lines)
    # left_lines=[line for line in lines if calculate_slope(line)<0]
    # print(left_lines)
    
    # lines1 = [right_lines[0],right_lines[2]]
    # lines2 = [right_lines[1],right_lines[3]]
    # lines3 = [left_lines[0],left_lines[2]]
    # lines4 = [left_lines[1],left_lines[3]]
    line1 = [[lines[0][0][0],lines[0][0][1]],[lines[0][0][2],lines[0][0][3]]]
    line2 = [[lines[1][0][0],lines[1][0][1]],[lines[1][0][2],lines[1][0][3]]]
    line3 = [[lines[2][0][0],lines[2][0][1]],[lines[2][0][2],lines[2][0][3]]]
    line4 = [[lines[3][0][0],lines[3][0][1]],[lines[3][0][2],lines[3][0][3]]]
    
    
    # '''4.离群值过滤'''
    # reject_abnormal_lines(right_lines,threshold=0.01)
    # reject_abnormal_lines(left_lines,threshold=0.01)

    '''5.最小二乘拟合 把识别到的多条线段拟合成一条直线'''
      #np.ravel: 将高维数组拉成一维数组
    # np.polyfit:多项式拟合
    #np.polyval: 多项式求值
    
    # line1 = least_squares_fit(lines1)  # *
    # line2 = least_squares_fit(lines2)
    # line3 = least_squares_fit(lines3)  
    # line4 = least_squares_fit(lines4)  # *
    
    print('line_points in ori_img',end=' : ')
    print(line1[0],line1[1],line4[0],line4[1])
    draw(img0,line1)
    # draw(img0,line2,2)
    # draw(img0,line3,3)
    draw(img0,line4,(0,255,0))
    cv2.namedWindow('ori_img_x',0)
    cv2.resizeWindow('ori_img_x',500,500)
    cv2.imshow('ori_img_x',img0)
    
    
    '''6.img2bev '''
    # pts1 = np.float32([[566,1079],[1530,1079],[1261,634],[784,639]]) # ori_img
    # pts2 = np.float32([[566,1079],[1530,1079],[1530,700],[566,700]]) # bev_img
    pts1 = np.float32([[775,631],[2101,612],[1349,320],[935,328]]) # ori_img
    pts2 = np.float32([[0,850],[1919,850],[1919,0],[0,0]]) # bev_img
    M = cv2.getPerspectiveTransform(pts1, pts2) # Matrix of ori_img -> bev_img
    img_bev = cv2.warpPerspective(img0, M, (img0.shape[1], img0.shape[0]), cv2.INTER_LINEAR)
    
    l1_p1, l1_p2 = perspectivePoint(line1, M) 
    l2_p1, l2_p2 = perspectivePoint(line4, M)
    print('line_points in bev_img',end=' : ')
    print(l1_p1, l1_p2, l2_p1, l2_p2)
    
    draw(img_bev,[l1_p1, l1_p2])
    draw(img_bev,[l2_p1, l2_p2],(0,255,0))
    cv2.namedWindow('bev_img_x',0)
    cv2.resizeWindow('bev_img_x',500,500)
    cv2.imshow('bev_img_x',img_bev)
    
    
    '''7.calculate the scale '''
    pixel_distance = get_distance_point2line(l1_p1,[l2_p1[0],l2_p1[1],l2_p2[0],l2_p2[1]])
    lane_width = 3.6  # unit:m  60km/h~3.5m ,>60km/h~3.75m
    scale = lane_width / pixel_distance
    print('Lane width in bev_img is <{}> pixel'.format(round(pixel_distance)))
    print('Lane width in real world is <{}> m'.format(lane_width))
    print('A pixel means <{:6f}> m'.format(scale))

def find_scale_y():
    '''1.canny边缘检测'''
    img_path = '/home/zxc/catkin_ws/src/camera/scripts/yutong.jpg'
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)     #以灰度图形式读取图片，为canny边缘检测做准备
    img0=cv2.imread(img_path,cv2.IMREAD_COLOR)
    
    edge_img=cv2.Canny(img,270, 240)     #设定阈值，低于阈值被忽略，高于阈值被显示，
                                           # 阈值的设定与图片的色彩有关，需要手动调整到合适的值（使车道线清晰显示出来）
    # cv2.imshow('edge_img',edge_img)
    
    '''2.roi_mask(提取感兴趣的区域)'''
    mask=np.zeros_like(edge_img)   #变换为numpy格式的图片
    mask=cv2.fillPoly(mask,np.array([[[875,534],[919,534],[905,606],[855,606]]]),color=255)   #对感兴趣区域制作掩膜
    # cv2.namedWindow('mask',0)
    # cv2.resizeWindow('mask',800,1200)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    masked_edge_img=cv2.bitwise_and(edge_img,mask)   #与运算
    # cv2.imshow('mask',masked_edge_img)
    # cv2.waitKey(0)
    '''3.霍夫变换，找出直线'''
    
    lines=cv2.HoughLinesP(masked_edge_img,1,np.pi/180,15,minLineLength=1,maxLineGap=20)    #获取所有线段
    lines = [line for line in lines if abs(calculate_slope(line))<0.5]
    print(lines)
    # return 0
    # right_lines=[line for line in lines if calculate_slope(line)>0]
    # left_lines=[line for line in lines if calculate_slope(line)<0]
    
    # print("right:{}".format(right_lines))  # 4条
    # print("left:{}".format(left_lines))  # 4条
    # lines1 = [right_lines[0],right_lines[2]]
    # lines2 = [right_lines[1],right_lines[3]]
    # lines3 = [left_lines[0],left_lines[2]]
    # lines4 = [left_lines[1],left_lines[3]]
    
    # '''4.离群值过滤'''
    # reject_abnormal_lines(right_lines,threshold=0.01)
    # reject_abnormal_lines(left_lines,threshold=0.01)

    '''5.最小二乘拟合 把识别到的多条线段拟合成一条直线'''
      #np.ravel: 将高维数组拉成一维数组
    # np.polyfit:多项式拟合
    #np.polyval: 多项式求值
    
    # line1 = least_squares_fit(lines1)  # *
    # line2 = least_squares_fit(lines2)
    # line3 = least_squares_fit(lines3)  
    # line4 = least_squares_fit(lines4)  # *
    
    line1 = [[lines[0][0][0],lines[0][0][1]],[lines[0][0][2],lines[0][0][3]]]
    line2 = [[lines[1][0][0],lines[1][0][1]],[lines[1][0][2],lines[1][0][3]]]
    
    print('line_points in ori_img',end=' : ')
    print(line1[0],line1[1],line2[0],line2[1])
    draw(img0,line1,tl=1)
    # draw(img0,line2,2)
    # draw(img0,line3,3)
    draw(img0,line2,(0,255,0),tl=1)
    cv2.namedWindow('ori_img_y',0)
    cv2.resizeWindow('ori_img_y',500,500)
    cv2.imshow('ori_img_y',img0)
    
    
    '''6.img2bev '''
    pts1 = np.float32([[775,631],[2101,612],[1349,320],[935,328]]) # ori_img
    pts2 = np.float32([[0,850],[1919,850],[1919,0],[0,0]]) # bev_img
    M = cv2.getPerspectiveTransform(pts1, pts2) # Matrix of ori_img -> bev_img
    img_bev = cv2.warpPerspective(img0, M, (img0.shape[1], img0.shape[0]), cv2.INTER_LINEAR)
    
    l1_p1, l1_p2 = perspectivePoint(line1, M) 
    l2_p1, l2_p2 = perspectivePoint(line2, M)
    print('line_points in bev_img',end=' : ')
    print(l1_p1, l1_p2, l2_p1, l2_p2)
    
    draw(img_bev,[l1_p1, l1_p2],tl=1)
    draw(img_bev,[l2_p1, l2_p2],(0,255,0),tl=1)
    cv2.namedWindow('bev_img_y',0)
    cv2.resizeWindow('bev_img_y',500,500)
    cv2.imshow('bev_img_y',img_bev)
    
    
    '''7.calculate the scale '''
    pixel_distance = get_distance_point2line(l1_p1,[l2_p1[0],l2_p1[1],l2_p2[0],l2_p2[1]])
    lane_width = 4.8  # unit:m             pedestrain lane min width is 3 meter.
    scale = lane_width / pixel_distance
    print('Lane width in bev_img is <{}> pixel'.format(round(pixel_distance)))
    print('Lane width in real world is <{}> m'.format(lane_width))
    print('A pixel means <{:6f}> m'.format(scale))

if __name__ == '__main__':
    print('--------scale_x--------')
    find_scale_x()
    print('--------scale_y--------')
    find_scale_y()
    cv2.waitKey(0)
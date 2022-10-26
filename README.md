# catkin_ws

#### 介绍
{**宇通车路协同项目视觉感知部分**}

#### 软件架构
[yolov4-darknet](https://https://github.com/AlexeyAB/darknet)

[deepsort](https://blog.csdn.net/didiaopao/article/details/120274519?spm=1001.2014.3001.5502)

[yolov5-6](https://github.com/ultralytics/yolov5)

[deep-text-recognition-benchmark](https://https://github.com/clovaai/deep-text-recognition-benchmark)

ROS

[yolov4-trt](https://github.com/jkjung-avt/tensorrt_demos)

[yolov5-trt](https://github.com/wang-xinyu/tensorrtx)


#### 安装教程

1.  catkin_make
2.  添加环境变量到 ~/.bashrc
3.  pip install -r requirement.txt

#### 使用说明

1. roscore 

2. rosrun camear cvsort.py

3. rosrun camear lpsort.py

4. roslaunch multi_object_tracker multi_object_tracker.launch
### #error
如果提示找不到yolosort，则把/devel/lib/*.py 全删了，不知道还有没有其他办法


#### 更新信息
 // version 1 (branch main)
1.  -2022/08/03 version 1.0
    节点1做车辆目标检测及跟踪，节点2做车牌检测及OCR并发布车牌信息给节点1
2.  -2022/08/04 version 1.1
    在上一版基础上修改如下：
    
    （1）优化detection.car，存储图片-->存储坐标，便于对ROI区域进行判断，写了接口，后续可以扩展；
    
    （2）增加了publisher，用来发布每一帧bbox的结果，格式list of [class, cx, cy, w, h, id, lp]，不过class目前都归类为‘Car’; 
    
    （3）将torch.Tensor改为numpy，但显存没有降低。
3.  -2022/08/08 version 1.2
    用CCPD2019、CCPD2020、CLPD混合数据集重新训练OCR模型，改用'None-VGG-BiLSTM-CTC'model，单张推理时间在2-3ms，权重仅34MB，精度为92.9%
    8574/1558 + 9939/1837 + 1100/100
    ('None-ResNet-BiLSTM-CTC'model，单张推理时间在5-6ms，权重198MB，精度为94.9%）
 
 // version 2 (branch version2)
1.  -2022/08/08 version 2.0
    合并为单个node, 目前帧率在15fps左右
2.  -2022/08/11 version 2.1
    修改了输入方式，现在可以接收别的node发送过来的图片了
    重写了一下主函数，更简洁了。
3.  -2022/08/13 version 2.2
    用sort替代deepsort, 暂时取消了lp检测和识别部分
    
 // version 3.0 version 3.0
1.  -2022/08/22 version 3.1
    (1)把yolov4-tiny的darknet模型转换为tensorrt模型，推理速度提升(448fps -> 1131fps，在3080显卡仅网络推理)，重写了objdetector.py文件
    
       参考[yolov4-trt](https://github.com/jkjung-avt/tensorrt_demos) 
       
    (2)修改了bbox的消息格式，与莫工提供的消息格式保持统一
    
       /msg/Boundingbox.msg   /msg/Boundingboxes.msg 

2.  -2022/08/25 version 3.2
    (1)sort现在可以保存box的类别了，实现多类别跟踪，并输出不同颜色的框
    
    (2)在bev图像中测量车道线距离，确定图像像素与真实世界坐标系之间的比例
        具体使用方法：
        先运行get_canny_parameters.py，通过拖动滑动条来调整canny边缘检测阈值，
        再运行get_scale.py，需要手动调整ROI区域，来定位相邻车道线和人行横道宽度，在bev中测量车道线的像素距离来求比例尺
        （只需要运行一次即可）
    
    (3)更新了图像，可以生成bev图像了。
    
3.  -2022/09/19 version 3.3
    (0)前面的更新都忘了上传哈哈
    
    (1)sort改名camera，msg不再使用，消息改用perception_msg
    
    (2)不再使用cv_bridge（在agx上编译太麻烦惹），用ros_numpy代替，好东西！
    
    (3)不再使用cv2.imshow做可视化，将图像以消息发送，可以用rviz接收并显示
    
    (4)更新了许多杂七杂八的东西...
    
    待更新：
    
    车牌检测及识别加到跟踪中，目前车牌检测用yolov5s-tensorrt(int8),识别用'None-VGG-BiLSTM-CTC'-tensorrt(fp16)。
    在RTX3080上检测识别一张车牌时间在2ms以内。
    
4.  -2022/10/19 version 3.4 final
    (1)重新用cv_bridge替换ros_numpy，速度提高七八倍，直接在工作空间下编译没问题，若想在所有环境都编译成功问题较大。具体见[csdn](https://blog.csdn.net/LoveJSH/article/details/126942789?spm=1001.2014.3001.5502)    
        
    (2)sort中用的kalmanfilter改用deepsort自带的.py文件，提速效果显著。具体见[csdn](https://blog.csdn.net/LoveJSH/article/details/127125028)
    
    (3)sort中启用子线程做车牌检测和识别。具体见[csdn](https://blog.csdn.net/LoveJSH/article/details/127152536)
    
    (4)可视化部分修改了颜色和文本位置，改用pillow-smid可显示中文。具体见[csdn](https://blog.csdn.net/LoveJSH/article/details/127287032)
    
    (5)添加了.launch文件，可以一键启动两个节点了: *roslaunch camera camera.launch*
    
    其他基本没啥了，项目结项就以这个作为最终版。
    
    待解决：
    
    2Dbbox接地点无法准确反映目标位置，可能得用3Dbbox来解决。
    
    由于小目标检测很容易漏检，所以跟踪时效果不佳；透视视角近大远小，大车容易遮挡小车，对跟踪影响较大。考虑在跟踪方面进行优化。
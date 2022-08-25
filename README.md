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

1.  roscore 
2.  rosrun deepsort lp_process.py        # 车牌检测及识别node
3.  rosrun deepsort yolosort.py          # 车辆检测及deepsort

### #error
如果提示找不到yolosort，则把/devel/lib/*.py 全删了，不知道还有没有其他办法

### 问题
OCR目前的权重是CCPD2020数据集训练的，所以结果都是新能源车牌且区域码都是安徽
目前占用显存比较多，车牌检测及OCR结点大约占用4GB，yolov4及deepsort结点大约占5GB

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
    (1)把yolov4-tiny的darknet模型转换为tensorrt模型，推理速度平均282fps，重写了objdetector.py文件
    
       参考[yolov4-trt](https://github.com/jkjung-avt/tensorrt_demos) 
       
    (2)修改了bbox的消息格式，与莫工提供的消息格式保持统一
    
       /msg/Boundingbox.msg   /msg/Boundingboxes.msg 

2.  -2022/08/25 version 3.2
    (1)sort现在可以保存box的类别了，实现多类别跟踪，并输出不同颜色的框
    
    (2)在bev图像中测量车道线距离，确定图像像素与真实世界坐标系之间的比例
        具体使用方法：
        先运行get_canny_parameters.py，通过拖动滑动条来调整canny边缘检测阈值，
        再运行get_scale.py，需要手动调整ROI区域，来定位相邻车道线，在bev中测量车道线的像素距离来求比例尺
        （只需要运行一次即可）
    
    (3)更新了图像，可以生成bev图像了。
        
        
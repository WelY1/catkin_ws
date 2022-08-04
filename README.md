# catkin_ws

#### 介绍
{**宇通车路协同项目视觉感知部分**}

#### 软件架构
[yolov4-darknet](https://https://github.com/AlexeyAB/darknet)

[deepsort](https://blog.csdn.net/didiaopao/article/details/120274519?spm=1001.2014.3001.5502)

[yolov5-6](https://github.com/ultralytics/yolov5)

[deep-text-recognition-benchmark](https://https://github.com/clovaai/deep-text-recognition-benchmark)

ROS


#### 安装教程

1.  catkin_make
2.  添加环境变量到 ~/.bashrc
3.  pip install -r requirement.txt

#### 使用说明

1.  roscore
2.  rosrun deepsort lp_process.py
3.  rosrun deepsort yolosort.py

### #error
如果提示找不到yolosort，则把/devel/lib/*.py 全删了，不知道还有其他办法没有

#### 更新信息

1.  -2022/08/03 version 1.0
    节点1做车辆目标检测及跟踪，节点2做车牌检测及OCR并发布车牌信息给节点1
2.  -2022/08/04 version 1.1
    在上一版基础上修改如下：
    
    （1）优化detection.car，存储图片-->存储坐标，便于对ROI区域进行判断，写了接口，后续可以扩展；
    
    （2）增加了publisher，用来发布每一帧bbox的结果，格式list of [class, cx, cy, w, h, id, lp]，不过class目前都归类为‘Car’; 
    
    （3）将torch.Tensor改为numpy，但显存没有降低。

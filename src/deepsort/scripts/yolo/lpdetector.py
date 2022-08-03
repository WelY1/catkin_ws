import torch
import numpy as np

from rospkg import RosPack

# 需要添加一个路径
import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.dataloaders import letterbox
from utils.torch_utils import select_device

DETECTOR_PATH = RosPack().get_path('image_trans')+'/scripts/yolo/weights/best.pt'




class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.img_size = 640
        self.confthreshold = 0.3
        self.iouthreshold = 0.4
        self.stride = 1
        

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, device=self.device)
        model.to(self.device).eval()
        model.half()
        self.net = model

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img                    # img0是原图，img是预处理后的图

    def detect(self, im):
        im0, img = self.preprocess(im)
        pred = self.net(img, augment=False)[0]     #pred: tensor
        pred = pred.float()
        pred = non_max_suppression(pred, self.confthreshold, self.iouthreshold)        # NMS  
        
        maxconf = 0
        lpbox = ()
        if len(pred):
            for det in pred:
                if len(det):                
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()         # 把img的box的尺寸还原到原图尺寸，先padding再缩放
                    for *x, conf, _ in det:                # det:[N, 6]      [x1, y1, x2, y2, confidence, label]
                        if conf > maxconf:
                            x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
                            maxconf = conf
                            lpbox = (x1,y1,x2,y2)
        return lpbox  



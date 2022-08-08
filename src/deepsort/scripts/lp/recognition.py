import numpy as np
import cv2

import string
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .model import Model

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import  AlignCollate2

from rospkg import RosPack

'''
封装的文本识别的部分
INPUT：im_crops  # list of lp_img
OUTPUT：lps          # list of lp_str
'''

class Recognition(object):
    def __init__(self, use_cuda=True):
        self.workers = 4
        self.batch_size = 64
        self.saved_model = RosPack().get_path('deepsort') + "/scripts/lp/None-VGG-BiLSTM-CTC.pth"
        """ Data processing """
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = True
        self.PAD = True
        self.sensitive = False
        self.character = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ藏川鄂甘赣港贵桂黑沪吉冀津晋京辽鲁蒙闽宁青琼陕苏皖湘新渝豫粤云浙'
        """ Model Architecture """
        self.Transformation = 'None'    # 'TPS'
        self.FeatureExtraction = 'VGG' # 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'CTC'   # 'Attn'
        self.num_fiducial = 20
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256

        if 'CTC' in self.Prediction:
            self.converter = CTCLabelConverter(self.character)
        else:
            self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)

        """ vocab / character number configuration """
        if self.sensitive:
            self.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.num_gpu = torch.cuda.device_count()
        self.net = Model(
                        self.batch_max_length,
                        self.imgH,
                        self.imgW,
                        self.Transformation,
                        self.FeatureExtraction,
                        self.SequenceModeling,
                        self.Prediction,
                        self.num_fiducial,
                        self.input_channel,
                        self.output_channel,
                        self.hidden_size,
                        self.num_class
                        )
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net = torch.nn.DataParallel(self.net).to(self.device)
        self.net.load_state_dict(torch.load(self.saved_model, map_location=self.device))
        
        self.size = (self.imgH, self.imgW)

        self.AlignCollate_demo = AlignCollate2(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD) #resize


# __call__()是一个非常特殊的实例方法。该方法的功能类似于在类中重载 () 运算符，
# 使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, im_crops):
        self.net.eval()
        im_batch = self.AlignCollate_demo(im_crops)          # 与训练时一样的预处理方式
        lps = []
        confss = []
        with torch.no_grad():
            batch_size = im_batch.size(0)
            image = im_batch.to(self.device)
            # For max length prediction
            length_for_pred = torch.IntTensor([self.batch_max_length]* batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(self.device)

            if 'CTC' in self.Prediction:
                preds = self.net(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)s
                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.net(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if 'Attn' in self.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                          # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()           # item()把tensor类型转换为python类型
            
        return pred, confidence_score               # lps返回的是string


if __name__ == '__main__':
    
    recognition =  Recognition()
    
    for i in range(12):
    
        img = []
        file_name = "/home/zxc/catkin_ws/src/deepsort/scripts/lp/demo_image/" + str(i+1) + ".jpg"
        im = cv2.imread(file_name)
        img.append(im)
        
        start = time.time()
        
        lp, conf =recognition(img)
        print(lp, conf, time.time()-start)


import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from PIL import Image

from .model import Model

import string
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F


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
        self.batch_size = 192
        self.saved_model = RosPack().get_path('deepsort') + "/scripts/lp/best_accuracy.pth"
        """ Data processing """
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = True
        self.PAD = False
        self.sensitive = False
        self.character = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ藏川鄂甘赣港贵桂黑沪吉冀津晋京辽鲁蒙闽宁青琼陕苏皖湘新渝豫粤云浙'
        """ Model Architecture """
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
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

        self.net = Model(self.num_class)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net = torch.nn.DataParallel(self.net).to(self.device)
        self.net.load_state_dict(torch.load(self.saved_model, map_location=self.device))
        
        self.size = (32, 100)
        # self.norm = transforms.Compose([
        #     # RGB图片数据范围是[0-255]，需要先经过ToTensor除以255归一化到[0,1]之后，
        #     # 再通过Normalize计算(x - mean)/std后，将数据归一化到[-1,1]。
        #     transforms.ToTensor(),
        #     # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]是从imagenet训练集中算出来的
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        self.AlignCollate_demo = AlignCollate2(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD)


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im, size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch
    
    

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
                    # index = []
                    # for ch in pred:
                    #     ch_index = self.character.find(ch)
                    #     index.append(ch_index)
                    # # pred = pred.replace('o','0').replace('i','1').upper() # 车牌号无“o”和“i”,全部变成大写
                    # pred_max_prob = pred_max_prob[:pred_EOS]
                    # lps.append(index)
                    lps.append(pred)

                          # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()           # item()把tensor类型转换为python类型
                confss.append(confidence_score)

        return lps[0], float(confss[0])                 # lps返回的是string


if __name__ == '__main__':
    img = []
    for i in range(12):
        file_name = f"demo_image/" + str(i+1) + ".jpg"
        # im =  Image.open(file_name)
        im = cv2.imread(file_name)
        img.append(im)
    recognition =  Recognition()
    lps, conf =recognition(img)
    lp = zip(lps, conf)
    for lpid, cof in lp:
        print(lpid, cof)


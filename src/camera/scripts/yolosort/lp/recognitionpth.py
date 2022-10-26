import numpy as np
import cv2

import time
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import math

from .model import Model

import torchvision.transforms as transforms
from rospkg import RosPack


'''
封装的文本识别的部分
INPUT：im_crops  # list of lp_img
OUTPUT：lps          # list of lp_str
'''

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img

class AlignCollate2(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        images = batch

        resized_max_w = self.imgW
        input_channel = 3
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.shape[1],image.shape[0]
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)
            resized_image = cv2.resize(image,(resized_w, self.imgH), cv2.INTER_CUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        return image_tensors

class Recognition(object):
    def __init__(self, use_cuda=True):
        self.saved_model = RosPack().get_path('camera') + '/scripts/yolosort/lp/None-VGG-BiLSTM-CTC.pth'
        """ Data processing """
        self.imgH = 32
        self.imgW = 100
        self.PAD = True
        self.character = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ藏川鄂甘赣港贵桂黑沪吉冀津晋京辽鲁蒙闽宁青琼陕苏皖湘新渝豫粤云浙'
        """ Model Architecture """
        self.converter = CTCLabelConverter(self.character)
        
        cudnn.benchmark = True
        cudnn.deterministic = True
        self.net = Model()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        # Only one GPU can be used to export onnx, so comment out the following line.
        # self.net = torch.nn.DataParallel(self.net).to(self.device)  
        self.net = self.net.to(self.device)
        self.net.load_state_dict(torch.load(self.saved_model, map_location=self.device))
        self.AlignCollate_demo = AlignCollate2(imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD) #resize
        
    
    def infer(self, im_crops):
        start21 = time.time()
        im_batch = self.AlignCollate_demo(im_crops)          # 与训练时一样的预处理方式
        time21 = time.time() - start21
        # print(type(im_batch))
        # print(im_batch.shape)
        with torch.no_grad():
            batch_size = im_batch.size(0)
            image = im_batch.to(self.device)   # torch.size([1,3,32,100])
            # For max length prediction
            start22 = time.time()
            preds = self.net(image)  # torch.Size([1, 24, 67])
            time22 = time.time() - start22
            start23 = time.time()
            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)s
            preds_str = self.converter.decode(preds_index, preds_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
                # calculate confidence score (= multiply of pred_max_prob)
            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
             # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            time23 = time.time() - start23
            if pred[1] in ['0','1','2','3','4','5','6','7','8','9']:
                pred = '豫A' + pred[2:]
            elif pred[0] != '豫': 
                pred = '豫' + pred[1:]
            while len(pred) < 7:
                pred = pred + str(self.character[np.random.randint(0,34)])
        return pred           # lp
        
        
    def infer2(self, im_crops):
        pred = '豫A'
        while len(pred) < 7:
            pred = pred + str(self.character[np.random.randint(0,34)])
        return pred


def warmup():
    # warmup
    image_dir = "/home/zxc/yutong/lp_det_ocr_TensorRT/demo_image/*.jpg"
    imgs = glob.glob(image_dir)
    for img in imgs:
        start24 = time.time()
        im = cv2.imread(img)
        lp, _,time21,time22,time23 = recognition.infer([im])
        
def test():
    avg21 = 0
    avg22 = 0
    avg23 = 0
    avg24 = 0
    image_dir = "/home/zxc/yutong/lp_det_ocr_TensorRT/demo_image/*.jpg"
    imgs = glob.glob(image_dir)
    for img in imgs:
        start24 = time.time()
        im = cv2.imread(img)
        lp, _,time21,time22,time23 = recognition.infer([im])
        time24 = time.time() - start24
        print('----------------------------------------------------')
        avg21 = time21 if avg21==0 else avg21 * 0.95 + time21 * 0.05   # ocr preprocess
        avg22 = time22 if avg22==0 else avg22 * 0.95 + time22 * 0.05   # ocr net infer
        avg23 = time23 if avg23==0 else avg23 * 0.95 + time23 * 0.05   # ocr postprecess
        avg24 = time24 if avg24==0 else avg24 * 0.95 + time24 * 0.05   # ocr total
        print('ocr preprocess          : {:.2f} ms'.format(avg21 * 1000))
        print('ocr net infer           : {:.2f} ms'.format(avg22 * 1000))
        print('ocr postprecess         : {:.2f} ms'.format(avg23 * 1000))
        print('ocr total               : {:.2f} ms'.format(avg24 * 1000))

def torch2onnx(model, onnx_path):
        model.eval()
        x = torch.randn(1, 3, 32, 100, device='cuda')
        print(x.shape)
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(
            model,
            x,
            onnx_path,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11 
        )
        print('->>模型转换成功！')

if __name__ == '__main__':
    
    recognition =  Recognition()
    warmup()
    test()
    # torch2onnx(recognition.net, 'ocr.onnx')
    
    
    


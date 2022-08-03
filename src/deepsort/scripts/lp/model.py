"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()
        self.workers = 4
        self.batch_size = 192
        self.saved_model = 'best_accuracy.pth'
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = True
        self.PAD = False
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        self.stages = {'Trans': self.Transformation, 'Feat': self.FeatureExtraction,
                       'Seq': self.SequenceModeling, 'Pred': self.Prediction}
        self.num_class = num_class

        """ Transformation """
        if self.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=self.num_fiducial, I_size=(self.imgH, self.imgW), I_r_size=(self.imgH, self.imgW), I_channel_num=self.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if self.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(self.input_channel, self.output_channel)
        elif self.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(self.input_channel, self.output_channel)
        elif self.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(self.input_channel, self.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = self.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if self.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size, self.hidden_size),
                BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))
            self.SequenceModeling_output = self.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if self.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, self.num_class)
        elif self.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, self.hidden_size, self.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.batch_max_length)

        return prediction

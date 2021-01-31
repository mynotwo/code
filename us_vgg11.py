import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg

from .slimmable_ops import USBatchNorm2d, USConv2d, make_divisible
from utils.config import FLAGS

cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '11-wide': [96, 'M', 192, 'M', 384, 384, 'M', 768, 768, 'M', 768, 768, 'M'],
}

class Model(nn.Module):
    def __init__(self, num_classes=200, input_size=64):
        super(Model, self).__init__()
        config = cfgs['11-wide']
		# enable dropout
        self.features = self.make_layers(config, True, True)
		# disable dropout
		# self.features = self.make_layers(config, True, False)
        self.dense = nn.Linear(int(config[-2])*4, num_classes)

        if FLAGS.reset_parameters:
            self.reset_parameters()

    def make_layers(self, cfg, batch_norm=False, drop_out=True):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if idx == 0:
                    conv2d = USConv2d(in_channels, v, 3, 1, groups=1, padding=1,
                                      depthwise=False, bias=True, us=[False, True], ratio=[1, 1])
                    bn_layer = USBatchNorm2d(v)
                elif idx == 11:
                    conv2d = USConv2d(in_channels, v, 3, 1, groups=1, padding=1,
                                      depthwise=False, bias=True, us=[True, False], ratio=[1, 1])
                    bn_layer = nn.BatchNorm2d(v)
                else:
                    conv2d = USConv2d(in_channels, v, 3, 1, groups=1, padding=1,
                                      depthwise=False, bias=True, ratio=[1, 1])
                    bn_layer = USBatchNorm2d(v)

                if batch_norm:
					layers += [conv2d, bn_layer, nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
				
				if drop_out:
					layers += [nn.Dropout(p=0.3)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print('-----')
        # print(x.shape)
        # print(self.dense.weight.shape)
        # print(out.shape)
        out = self.dense(out)
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
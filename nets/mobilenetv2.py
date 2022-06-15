#---------------------------------------------------#
#   MobileNet-V2: 倒残差结构
#   1x1 3x3DWConv 1x1
#---------------------------------------------------#

import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url

BatchNorm2d = nn.BatchNorm2d

# 卷积,标准化,激活函数
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 1x1卷积,标准化,激活函数
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

#---------------------------------------------------#
#   倒残差结构
#   残差:   两端channel多,中间channel少
#       降维 --> 升维
#   倒残差: 两端channel少,中间channel多
#       升维 --> 降维
#   1x1 3x3DWConv 1x1
#   最后的1x1Conv没有激活函数
#---------------------------------------------------#
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        # 步长为1同时通道不变化才相加
        self.use_res_connect = self.stride == 1 and inp == oup
        #----------------------------------------------------#
        #   利用1x1卷积根据输入进来的通道数进行通道数上升,不扩张就不需要第一个1x1卷积了
        #----------------------------------------------------#
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #----------------------------------------------------#
                #   利用深度可分离卷积进行特征提取
                #----------------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #----------------------------------------------------#
                #   利用1x1的卷积进行通道数的下降,没有激活函数
                #----------------------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),   # 没有激活函数
            )
        else:
            self.conv = nn.Sequential(
                #----------------------------------------------------#
                #   利用1x1卷积根据输入进来的通道数进行通道数上升
                #----------------------------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #----------------------------------------------------#
                #   利用深度可分离卷积进行特征提取
                #----------------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #----------------------------------------------------#
                #   利用1x1的卷积进行通道数的下降
                #----------------------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),   # 没有激活函数
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # 扩张,out_channel,重复次数,stride
            # t, c, n, s
            #                 高,宽,通道
            [1, 16, 1, 1],  # 237,237,32 -> 237,237,16
            [6, 24, 2, 2],  # 237,237,16 -> 119,119,24  开始id:2
            [6, 32, 3, 2],  # 119,119,24 -> 60,60,32    开始id:4

            # 后面两层的膨胀系数会根据downsample_factor调整, 为8这里为2
            [6, 64, 4, 2],  # 60,60,32 -> 30,30,64          开始id:7    stride会根据downsample_factor调整, 为16这里为2, 为8这里为1
            [6, 96, 3, 1],  # 30,30,64 -> 30,30,96  # 这个输出为aux层输出

            # 后面两层的膨胀系数会根据downsample_factor调整, 为16这里为2, 为8这里为4
            [6, 160, 3, 2], # 30,30,96 -> 15,15,160     开始id:14       stride会变为1,保持高宽为30
            [6, 320, 1, 1], # 15,15,160 -> 15,15,320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 第一层卷积 473,473,3 -> 237,237,32
        self.features = [conv_bn(3, input_channel, 2)]

        # 根据上述列表进行循环，构建mobilenetv2的结构
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                # 第一次步长为2,其余为1
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # PSPNet不会使用后面的分类块,只进行特征提取
        # mobilenetv2结构的收尾工作
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # 最后的分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url('https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar', "./model_data"), strict=False)
    return model

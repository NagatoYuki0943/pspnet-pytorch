"""
加强特征提取网络
对resnet和mobilenetv2进行修改
"""

import torch
import torch.nn.functional as F
from torch import nn

from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        #--------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1[0]
        self.bn1 = model.conv1[1]
        self.relu1 = model.conv1[2]
        self.conv2 = model.conv1[3]
        self.bn2 = model.conv1[4]
        self.relu2 = model.conv1[5]
        self.conv3 = model.conv1[6]
        self.bn3 = model.bn1
        self.relu3 = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        """
        m: apply中的一个模型实例
        dilate: apply中的自定义参数，扩张系数
        """
        # classname = m.__class__.__name__
        # if classname.find('Conv') != -1:  # != -1 表示找得到"Conv"
        # if type(m) == nn.Conv2d:
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # 返回两层
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x

class MobileNetV2(nn.Module):
    '''
    修改原本的mobilenetv2,不要最后的classifier和features的最后一层
    返回最后一层InvertedResidual和倒数第三层的输出作为aux
    '''
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1] # 不要features层最后的1x1Conv

        self.total_idx = len(self.features) # 18层
        self.down_idx = [2, 4, 7, 14]   # 下采样层的id开始序号

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        #--------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            # 3次下采样,后面的stride都改为1
            # 将mobilenetv2后三四层卷积的膨胀系数设为2
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            # 将mobilenetv2后两层卷积的膨胀系数设为4
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            # 4次下采样,最后一次stride改为1
            # 将mobilenetv2后两层卷积的膨胀系数设为2
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        '''调整dilate和stride'''
        classname = m.__class__.__name__
        # 修改卷积层
        if classname.find('Conv') != -1:
            # 将深层的步长由2改为1,减少一层下采样
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)     # 2/2=1 不进行扩张卷积
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # 返回两层
        # a_xux: [30,30,96]
        x_aux = self.features[:14](x)
        # x: [30,30,320]
        x = self.features[14:](x_aux)
        return x_aux, x


#-----------------------------------------------------------------------------#
#   PSPModule
#   获取到的特征层划分成不同大小的区域，每个区域内部各自进行平均池化。
#   实现聚合不同区域的上下文信息，从而提高获取全局信息的能力
#   最终将in_channels和四个特征区域的输出合并通道
#                                    in
#                                     │
#       ┌──────────────┬──────────────┼──────────────┬──────────────┐
#       │              │              │              │              │
#       │          AvgPool2d      AvgPool2d      AvgPool2d      AvgPool2d
#       │           1x1out         2x2out         3x3out         6x6out
#       │              │              │              │              │
#       │           1x1Conv        1x1Conv        1x1Conv        1x1Conv
#       │              │              │              │              │
#       │          UpSample       UpSample       UpSample       UpSample
#       │              │              │              │              │
#       │              └─────────────┐│┌─────────────┘              │
#       └─────────────────────────── cat ───────────────────────────┘
#                                     │
#                                  3x3Conv
#                                     │
#                                    out
#-----------------------------------------------------------------------------#
class _PSPModule(nn.Module):

    def __init__(self, in_channels, pool_sizes, norm_layer):    # pool_sizes = [1,2,3,6] 将特征划分为1x1,2x2,3x3,6x6的区域
        super(_PSPModule, self).__init__()
        # 80 = 320 / 4
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   循环进行设置不同的pool_size
        #-----------------------------------------------------#
        # 循环进行设置不同的kernel_size
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])

        # 合并通道,特征融合 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            # 320 + 80 * 4 = 640
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, pool_size, norm_layer):
        # 调整输出的高宽
        prior = nn.AdaptiveAvgPool2d(output_size=pool_size)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        # 高宽
        h, w = features.size(2), features.size(3)
        # 将输入放进列表中
        pyramids = [features]
        # 对输入的特征层进行重复计算,
        # F.interpolate 调整resize大小,让大小都变为初始的大小
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        # 拼接维度,调整维度,特征融合
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone=="resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        #--------------------------------------------------------------#
        #	PSP模块，分区域进行池化,对卷积结果进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   [30, 30, 320] -> [30, 30, 80] -> [30, 30, num_classes]
        #--------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            # 使用1x1卷积对PSPModule输入的数据进行特征提取即可  num_classes相当于对每个特征点进行分类
            # [30,30,80] => [30,30,num_classes]
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            #---------------------------------------------------#
            #	利用特征获得预测结果
            #   将卷积倒数第三层的输出进行处理
            #   [30, 30, 96] -> [30, 30, 40] -> [30, 30, num_classes]
            #---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size(2), x.size(3)) # [473,473]
        # 特征提取
        x_aux, x = self.backbone(x)
        # PSPNet [30,30,80] => [30,30,num_classes]
        output = self.master_branch(x)
        # 调整大小为输入图片大小 [30,30,80] => [473,473,num_classes]
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        # 辅助分支,没有PSPNet
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilemamba import MobileMamba_S6, MobileMamba_T2, MobileMamba_T4, MobileMamba_B1, MobileMamba_B2, MobileMamba_B4


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
"""
ASPP 类用于构建 Atrous Spatial Pyramid Pooling 模块，通过不同膨胀率的卷积提取多尺度特征。
参数:
    dim_in (int): 输入通道数。
    dim_out (int): 输出通道数。
    rate (int): 基础膨胀率，默认为 1。
    bn_mom (float): 批归一化动量，默认为 0.1。
"""


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    """
    forward 方法用于前向传播，通过五���分支提取特征并融合。
    参数:
        x (Tensor): 输入张量。
    返回:
        Tensor: 融合后的特征张量。
    """

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


"""
DeepLab 类用于构建 DeepLabv3+ 网络，结合了编码器-解码器结构和 ASPP 模块。
参数:
    num_classes (int): 分类数。
    backbone (str): 主干网络类���，支持 "mobilenet" 和 "xception"，默认为 "mobilenet"。
    pretrained (bool): 是否使用预训练权重，默认为 True。
    downsample_factor (int): 下采样因子，默认为 16。
"""


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilemamba", backbone_variant="S6", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        else:
            # 支持多种 MobileMamba 变体
            if backbone_variant == "S6":
                self.backbone = MobileMamba_S6(num_classes=0, pretrained=pretrained)
                in_channels = 448
                low_level_channels = 192
            elif backbone_variant == "T2":
                self.backbone = MobileMamba_T2(num_classes=0, pretrained=pretrained)
                in_channels = 368
                low_level_channels = 144
            elif backbone_variant == "T4":
                self.backbone = MobileMamba_T4(num_classes=0, pretrained=pretrained)
                in_channels = 448
                low_level_channels = 176
            elif backbone_variant == "B1":
                self.backbone = MobileMamba_B1(num_classes=0, pretrained=pretrained)
                in_channels = 448
                low_level_channels = 200
            elif backbone_variant == "B2":
                self.backbone = MobileMamba_B2(num_classes=0, pretrained=pretrained)
                in_channels = 448
                low_level_channels = 200
            elif backbone_variant == "B4":
                self.backbone = MobileMamba_B4(num_classes=0, pretrained=pretrained)
                in_channels = 448
                low_level_channels = 200
            else:
                raise ValueError(f"Unknown MobileMamba variant: {backbone_variant}")

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class MobileMamba(nn.Module):
    def __init__(self, backbone_variant="S6", downsample_factor=8, pretrained=True):
        super(MobileMamba, self).__init__()
        # 支持多种变体
        if backbone_variant == "S6":
            self.model = MobileMamba_S6(num_classes=0, pretrained=pretrained)
        elif backbone_variant == "T2":
            self.model = MobileMamba_T2(num_classes=0, pretrained=pretrained)
        elif backbone_variant == "T4":
            self.model = MobileMamba_T4(num_classes=0, pretrained=pretrained)
        elif backbone_variant == "B1":
            self.model = MobileMamba_B1(num_classes=0, pretrained=pretrained)
        elif backbone_variant == "B2":
            self.model = MobileMamba_B2(num_classes=0, pretrained=pretrained)
        elif backbone_variant == "B4":
            self.model = MobileMamba_B4(num_classes=0, pretrained=pretrained)
        else:
            self.model = MobileMamba_S6(num_classes=0, pretrained=pretrained)

    def forward(self, x):
        x = self.model.patch_embed(x)
        low_level_features = self.model.blocks1(x)
        x = self.model.blocks2(low_level_features)
        x = self.model.blocks3(x)
        return low_level_features, x

# 2490
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class SFM(nn.Module):
    def __init__(self):
        super(SFM, self).__init__()
        self.conv_f1 = nn.Conv2d(64, 64, 1)
        self.conv_f2 = nn.Conv2d(128, 64, 1)
        self.conv_f3 = nn.Conv2d(320, 64, 1)
        self.conv_f4 = nn.Conv2d(128, 64, 1)
        self.CovB = nn.Sequential(nn.MaxPool2d(2, 2),
                                  nn.Conv2d(512, 128, 1),
                                  nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
                                  )
        self.aspp = ASPP(128, 128)

    def forward(self, f1, f2, f3, f4):
        f4 = self.CovB(f4)
        f4 = self.aspp(f4)
        f4 = self.conv_f4(f4)
        up_f4 = F.interpolate(f4, f3.size()[2:], mode='bilinear')

        f3 = self.conv_f3(f3)
        f3 = f3 + up_f4
        f3 = self.conv_f1(f3)
        up_f3 = F.interpolate(f3, f2.size()[2:], mode='bilinear')

        f2 = self.conv_f2(f2)
        f2 = f2 + up_f3
        f2 = self.conv_f1(f2)
        up_f2 = F.interpolate(f2, f1.size()[2:], mode='bilinear')

        f1 = self.conv_f1(f1)
        f1 = f1 + up_f2

        return f1


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out

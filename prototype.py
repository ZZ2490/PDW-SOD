# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F

from Encoder import Encoder

import pdb


class Prototype(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self):
        super(Prototype, self).__init__()
        model = Encoder()
        self.backbone = model.encoder

        self.netup = torch.nn.Sequential(
            torch.nn.Conv2d(512, 24, 3, padding=1)
        )
        self.netup2 = torch.nn.Sequential(
            torch.nn.Conv2d(320, 24, 3, padding=1)
        )
        self.netup3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 24, 3, padding=1)
        )
        self.netup4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 24, 3, padding=1)
        )

        self.centroids = torch.nn.Parameter(torch.rand(24, 512))  # 生成（24, 256)可训练学习的张量
        self.centroids2 = torch.nn.Parameter(torch.rand(24, 320))
        self.centroids3 = torch.nn.Parameter(torch.rand(24, 128))
        self.centroids4 = torch.nn.Parameter(torch.rand(24, 64))

        self.num_cluster = 24

        self.upfc = torch.nn.Linear(24 * 512, 512)
        self.upfc2 = torch.nn.Linear(24 * 320, 320)
        self.upfc3 = torch.nn.Linear(24 * 128, 128)
        self.upfc4 = torch.nn.Linear(24 * 64, 64)

        self.transform = torch.nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.transform2 = torch.nn.Sequential(
            nn.Conv2d(640, 320, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(320, 320, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.transform3 = torch.nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.transform4 = torch.nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=False),
        )

    def UP(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)  # 对x做正则化，除2范数
        soft_assign = self.netup(x)  # 通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  # 通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        # 调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids  # 生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)  # 对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # 在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc(up).unsqueeze(2).unsqueeze(3).repeat(1, 1, W, H)

        return up, centroid

    def UP2(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)  # 对x做正则化，除2范数
        soft_assign = self.netup2(x)  # 通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  # 通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        # 调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids2  # 生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)  # 对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # 在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc2(up).unsqueeze(2).unsqueeze(3).repeat(1, 1, W, H)

        return up, centroid

    def UP3(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)  # 对x做正则化，除2范数
        soft_assign = self.netup3(x)  # 通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  # 通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        # 调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids3  # 生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)  # 对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # 在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc3(up).unsqueeze(2).unsqueeze(3).repeat(1, 1, W, H)

        return up, centroid

    def UP4(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)  # 对x做正则化，除2范数
        soft_assign = self.netup4(x)  # 通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  # 通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        # 调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids4  # 生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)  # 对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # 在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc4(up).unsqueeze(2).unsqueeze(3).repeat(1, 1, W, H)

        return up, centroid

    def forward(self, images):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        # images = to_image_list(images)
        src_features = self.backbone(images)

        up, centroid = self.UP(src_features[3])
        new_features = torch.cat((src_features[3], up), dim=1)
        new_features = self.transform(new_features)

        up1, centroid1 = self.UP2(src_features[2])
        new_features1 = torch.cat((src_features[2], up1), dim=1)
        new_features1 = self.transform2(new_features1)

        up2, centroid2 = self.UP3(src_features[1])
        new_features2 = torch.cat((src_features[1], up2), dim=1)
        new_features2 = self.transform3(new_features2)

        up3, centroid3 = self.UP4(src_features[0])
        new_features3 = torch.cat((src_features[0], up3), dim=1)
        new_features3 = self.transform4(new_features3)

        up_features = []
        up_features.append(new_features3)
        up_features.append(new_features2)
        up_features.append(new_features1)
        up_features.append(new_features)
        features = up_features

        return features

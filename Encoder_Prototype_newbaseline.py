import torch
import torch.nn as nn
import pvt_v2
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
import os, argparse
import cv2

from LWA import lwa, ReliabilitySelector, DQW, twolwa, DHA, MultiModalFusion
from SFM import SFM
from generalized_rcnn import GeneralizedRCNN
from prototype import Prototype
from prototype_w import prototype_w
from singe_prototype import Singe_prototype


# from model import convnext_small as create_model
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),  # 卷积块 卷积加批量标准化 激活
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class CA(nn.Module):  # 注意力机制
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)  # 平均池化 不改变通道数 只改变特征图大小
        self.max_weight = nn.AdaptiveMaxPool2d(1)  # 最大池化 不改变通道数 只改变特征图大小
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)


class CALayer(nn.Module):   #通道注意力机制
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),  # 1×1 卷积 通道数除16，卷积大小不变
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)  # 向下整除 步长

    def forward(self, x):    #残差连接的注意力块 加入了正则化和激活
        res = self.body(x)
        res += x
        return res


class CMF(nn.Module):
    def __init__(self, in_1, in_2):
        super(CMF, self).__init__()
        self.att = RCAB(in_1)
        self.conv_globalinfo = convblock(in_2, in_1, 1, 1, 0)
        self.p1 = Singe_prototype(in_1, 20)
        self.p2 = Singe_prototype(in_2, 20)
        self.gwm = MultiModalFusion(in_1, in_1)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_1, in_1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_fus = convblock(2 * in_1, in_1, 3, 1, 1)  #不改变图像尺寸，通道数减半
        self.conv_out = convblock(2 * in_1, in_1, 3, 1, 1)

    def forward(self, rgb, depth, global_info):
        rgb = self.p1(rgb)
        depth = self.p1(depth)
        rgb, depth = self.gwm(rgb, depth)
        global_info = self.p2(global_info)
        cur_size = rgb.size()[2:]  # 只保留RGB图像的大小 通道数和数量
        att_rgb = self.att(rgb)
        att_d = self.att(depth)         #注意力增强后的特征图 大小尺寸通道不变
        xd_in = att_d + att_rgb * att_d
        xr_in = att_rgb
        cmf_t = xd_in + torch.add(xd_in, torch.mul(xd_in, self.rt_fus(xr_in)))
        cmf_r = xr_in + torch.add(xr_in, torch.mul(xr_in, self.rt_fus(xd_in)))

        ful_mul = torch.mul(cmf_r, cmf_t)
        x_in1 = torch.reshape(cmf_r, [cmf_r.shape[0], 1, cmf_r.shape[1], cmf_r.shape[2], cmf_r.shape[3]])
        x_in2 = torch.reshape(cmf_t, [cmf_t.shape[0], 1, cmf_t.shape[1], cmf_t.shape[2], cmf_t.shape[3]])
        x_cat = torch.cat((x_in1, x_in2), dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul, ful_max), dim=1)
        cmf_out = self.conv_fus(ful_out)      #空间注意力的变种
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))

        return self.conv_out(torch.cat((cmf_out, global_info), 1))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sig = nn.Sigmoid()
        self.cmf1 = CMF(64, 128)
        self.cmf2 = CMF(128, 320)
        self.cmf3 = CMF(320, 512)
        self.cmf4 = CMF(512, 512)

    def forward(self, rgb_f, d_f):
        f_g = rgb_f[3] + d_f[3]
        f_4 = self.cmf4(rgb_f[3], d_f[3], f_g)
        f_3 = self.cmf3(rgb_f[2], d_f[2], f_4)
        f_2 = self.cmf2(rgb_f[1], d_f[1], f_3)
        f_1 = self.cmf1(rgb_f[0], d_f[0], f_2)

        return f_1, f_2, f_3, f_4, f_g


class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)


def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model


class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        model = Encoder()
        # self.rgb_net = model.encoder
        self.rgb_net = model.encoder
        self.t_net = model.encoder
        self.p1 = Singe_prototype(512, 20)
        self.p2 = Singe_prototype(320, 20)
        self.p3 = Singe_prototype(128, 20)
        self.p4 = Singe_prototype(64, 20)
        self.sfm = SFM()
        self.rs = ReliabilitySelector(128, 2)
        self.prototype = prototype_w(64, 20)
        self.lwa = lwa(64, 4)

        self.twolwa = twolwa(64, 4)
        self.dqw = DQW()
        self.conv_3 = convblock(320 * 2, 320, 3, 1, 1)
        self.conv_2 = convblock(128 * 2, 128, 3, 1, 1)
        self.cmf1 = CMF(64, 128)
        self.cmf2 = CMF(128, 320)
        self.cmf3 = CMF(320, 512)
        self.cmf4 = CMF(512, 512)
        self.decoder = Decoder()
        self.sig = nn.Sigmoid()
        self.dha = DHA()
        self.up_g = convblock(512, 64, 3, 1, 1)
        self.up_4 = convblock(512, 64, 3, 1, 1)
        self.up_3 = convblock(320, 64, 3, 1, 1)
        self.up_2 = convblock(128, 64, 3, 1, 1)
        self.up_1 = convblock(64, 64, 1, 1, 0)

        self.conv_s1 = convblock(64 * 5, 64, 3, 1, 1)
        self.conv_u1 = convblock(64 * 2, 64, 3, 1, 1)
        self.score1 = nn.Conv2d(64, 1, 1, 1, 0)

        self.conv_s2 = convblock(64 * 5, 64, 3, 1, 1)
        self.score2 = nn.Conv2d(64, 1, 1, 1, 0)

        self.score = nn.Conv2d(2, 1, 1, 1, 0)

    def forward(self, imgs, depths):
        img_1, img_2, img_3, img_4 = self.rgb_net(imgs)
        dep_1, dep_2, dep_3, dep_4 = self.t_net(depths)

        tha1, tha2, tha3, tha4 = self.dha(img_1, dep_1, dep_2, dep_3, dep_4)
        #
        # # rha1, rha2, rha3, rha4 = self.dha(dep_1, img_1, img_2, img_3, img_4)
        #
        # # img_f = self.sfm(img_1, img_2, img_3, img_4)
        # # dep_f = self.sfm(dep_1, dep_2, dep_3, dep_4)
        # #
        # # up1 = self.prototype(img_f)
        # # up2 = self.prototype(dep_f)
        # # # up_fuse = torch.cat((up1, up2), dim=1)
        # # #
        # # # up_w = self.rs(up_fuse).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # # # r_w = up_w[:, 0, ...]
        # # # d_w = up_w[:, 1, ...]
        # # r_w, d_w = self.gwm(up1, up2)
        # # img_4 = r_w * img_4
        # # img_3 = r_w * img_3
        # # img_2 = r_w * img_2
        # # img_1 = r_w * img_1
        #
        dep_4 = dep_4 * tha4
        dep_3 = dep_3 * tha3
        dep_2 = dep_2 * tha2
        dep_1 = dep_1 * tha1
        # # gate2 = self.lwa(up1, up2)
        #
        # dep_4 = tha4 * dep_4 * gate2[:, 3:4, ...]
        # dep_3 = tha3 * dep_3 * gate2[:, 2:3, ...]
        # dep_2 = tha2 * dep_2 * gate2[:, 1:2, ...]
        # dep_1 = tha1 * dep_1 * gate2[:, 0:1, ...]
        #
        # img_4 = rha4 * img_4 * gate1[:, 3:4, ...]
        # img_3 = rha3 * img_3 * gate1[:, 2:3, ...]
        # img_2 = rha2 * img_2 * gate1[:, 1:2, ...]
        # img_1 = rha1 * img_1 * gate1[:, 0:1, ...]


        # # print(up1.size())
        # # print(up2.size())

        # up_fuse = torch.cat((up1, up2), dim=1)
        # up_w = self.rs(up_fuse).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # r_w = up_w[:, 0, ...]
        # d_w = up_w[:, 1, ...]
        # print(r_w.size(), img_4.size())
        # print(r_w)
        # img_4 = r_w * img_4
        # img_3 = r_w * img_3
        # img_2 = r_w * img_2
        # img_1 = r_w * img_1
        #
        # dep_4 = d_w * dep_4
        # dep_3 = d_w * dep_3
        # dep_2 = d_w * dep_2
        # dep_1 = d_w * dep_1
        # w4 = self.lwa(up1, up2)

        # img_4 = self.p1(img_4)
        # img_3 = self.p2(img_3)
        # img_2 = self.p3(img_2)
        # img_1 = self.p4(img_1)
        #
        # dep_4 = self.p1(dep_4)
        # dep_3 = self.p2(dep_3)
        # dep_2 = self.p3(dep_2)
        # dep_1 = self.p4(dep_1)

        # dep_4 = w4[:, 3:4, ...] * dep_4
        # dep_3 = w4[:, 2:3, ...] * dep_3
        # dep_2 = w4[:, 1:2, ...] * dep_2
        # dep_1 = w4[:, 0:1, ...] * dep_1

        r_f_list = [img_1, img_2, img_3, img_4]
        d_f_list = [dep_1, dep_2, dep_3, dep_4]

        f1, f2, f3, f4, f_g = self.decoder(r_f_list, d_f_list)

        # f4 = self.p1(f4)
        # f3 = self.p2(f3)
        # f2 = self.p3(f2)
        # f1 = self.p4(f1)

        f_g = self.up_g(F.interpolate(f_g, f1.size()[2:], mode='bilinear', align_corners=True))
        f_4 = self.up_4(F.interpolate(f4, f1.size()[2:], mode='bilinear', align_corners=True))
        f_3 = self.up_3(F.interpolate(f3, f1.size()[2:], mode='bilinear', align_corners=True))
        f_2 = self.up_2(F.interpolate(f2, f1.size()[2:], mode='bilinear', align_corners=True))
        f_1 = self.up_1(f1)

        s_f = self.conv_s1(torch.cat((f_1, f_2, f_3, f_4, f_g), 1))
        # up_f = self.conv_u1(torch.cat((up1, up2), 1))
        # score_up = self.score1(F.interpolate(up_f, (384, 384), mode='bilinear', align_corners=True))
        score_f = self.score1(F.interpolate(s_f, (384, 384), mode='bilinear', align_corners=True))
        score_f2 = self.score1(F.interpolate(f1, (384, 384), mode='bilinear', align_corners=True))

        score = self.score(torch.cat((score_f + torch.mul(score_f, self.sig(score_f2)),
                                      score_f2 + torch.mul(score_f2, self.sig(score_f))), 1))

        return score, score_f, score_f2, self.sig(score)

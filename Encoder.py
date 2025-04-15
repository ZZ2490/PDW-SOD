import torch
import torch.nn as nn
import pvt_v2
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
import os, argparse
import cv2


# from model import convnext_small as create_model
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),  # 卷积块 卷积加批量标准化 激活
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class CA(nn.Module): #注意力机制
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


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),     # 1×1 卷积 通道数除16，卷积大小不变
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
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias) #向下整除 步长

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CMF(nn.Module):
    def __init__(self, in_1, in_2):
        super(CMF, self).__init__()
        self.att = RCAB(in_1)
        self.conv_globalinfo = convblock(in_2, in_1, 1, 1, 0)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_1, in_1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_fus = convblock(2 * in_1, in_1, 3, 1, 1)
        self.conv_out = convblock(2 * in_1, in_1, 3, 1, 1)

    def forward(self, rgb, depth, global_info):
        cur_size = rgb.size()[2:]    #只保留RGB图像的大小 通道数和数量
        att_rgb = self.att(rgb)
        att_d = self.att(depth)
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
        cmf_out = self.conv_fus(ful_out)
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))

        return self.conv_out(torch.cat((cmf_out, global_info), 1))


class R_CLCM(nn.Module):
    def __init__(self, in_1, in_2, in_3, in_4):
        super(R_CLCM, self).__init__()
        self.ca1 = CA(in_1)
        self.ca2 = CA(in_2)
        self.ca3 = CA(in_3)
        self.ca4 = CA(in_4)

        self.conv_r1 = convblock(in_1, in_1, 1, 1, 0)
        self.conv_r2 = convblock(in_2, in_1, 3, 1, 1)
        self.conv_r3 = convblock(in_3, in_1, 3, 1, 1)
        self.conv_r4 = convblock(in_4, in_1, 3, 1, 1)

        self.conv_n1 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n2 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n3 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)
        self.conv_n4 = nn.Conv2d(in_1, in_1 // 4, 1, 1, 0)

        self.conv_3_r = convblock(in_1 // 4, in_1, 1, 1, 0)
        self.conv_2_r = convblock(in_1 // 4, in_1, 1, 1, 0)
        self.conv_1_r = convblock(in_1 // 4, in_1, 1, 1, 0)

        self.conv_out1 = nn.Conv2d(in_1, in_1, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(in_1, in_2, 3, 1, 1)
        self.conv_out3 = nn.Conv2d(in_1, in_3, 3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gam3 = nn.Parameter(torch.zeros(1))
        self.gam2 = nn.Parameter(torch.zeros(1))
        self.gam1 = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, x3, x4):
        r1 = self.conv_r1(self.ca1(x1))
        r2 = self.conv_r2(self.ca2(x2))
        r3 = self.conv_r3(self.ca3(x3))
        r4 = self.conv_r4(self.ca4(x4))

        b, c, h1, w1 = r1.size()
        b, c, h2, w2 = r2.size()
        b, c, h3, w3 = r3.size()
        b, c, h4, w4 = r4.size()

        r_4 = self.conv_n4(r4).view(b, -1, h4 * w4)  # b, c, l4
        r_4_t = r_4.permute(0, 2, 1)  # b, l4, c
        r_3 = self.conv_n3(r3).view(b, -1, h3 * w3)  # b, c, l3

        r_4_3 = torch.bmm(r_4_t, r_3)  # b, l4, l3
        att_r_4_3 = self.softmax(r_4_3)
        r_3_4 = torch.bmm(r_4, att_r_4_3)  # b, c, l3
        r_3_in = r_3_4 + r_3  # b, c, l3

        r_3_in_t = r_3_in.permute(0, 2, 1)  # b, l3, c
        r_2 = self.conv_n2(r2).view(b, -1, h2 * w2)  # b, c, l2

        r_3_2 = torch.bmm(r_3_in_t, r_2)  # b, l3, l2
        att_r_3_2 = self.softmax(r_3_2)
        r_2_3 = torch.bmm(r_3_in, att_r_3_2)  # b, c, l2
        r_2_in = r_2_3 + r_2

        r_2_in_t = r_2_in.permute(0, 2, 1)  # b, l2, c
        r_1 = self.conv_n1(r1).view(b, -1, h1 * w1)  # b, c, l1

        r_2_1 = torch.bmm(r_2_in_t, r_1)  # b, l2, l1
        att_r_2_1 = self.softmax(r_2_1)
        r_1_2 = torch.bmm(r_2_in, att_r_2_1)  # b, c, l1
        r_1_in = r_1_2 + r_1

        r_3_out = self.conv_3_r(r_3_in.view(b, -1, h3, w3))
        out_r3 = self.gam3 * r_3_out + r3
        r_2_out = self.conv_2_r(r_2_in.view(b, -1, h2, w2))
        out_r2 = self.gam2 * r_2_out + r2
        r_1_out = self.conv_1_r(r_1_in.view(b, -1, h1, w1))
        out_r1 = self.gam1 * r_1_out + r1

        return self.conv_out1(out_r1), self.conv_out2(out_r2), self.conv_out3(out_r3)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sig = nn.Sigmoid()
        self.cmf1 = CMF(64, 128)
        self.cmf2 = CMF(128, 320)
        self.cmf3 = CMF(320, 512)
        self.cmf4 = CMF(512, 512)
        self.R_CLCM_1234 = R_CLCM(64, 128, 320, 512)
        self.R_CLCM_4321 = R_CLCM(512, 320, 128, 64)

        self.conv_3 = convblock(320 * 2, 320, 3, 1, 1)
        self.conv_2 = convblock(128 * 2, 128, 3, 1, 1)

    def forward(self, rgb_f, d_f):
        f_g = rgb_f[3] + d_f[3]
        f_4 = self.cmf4(rgb_f[3], d_f[3], f_g)
        f_3 = self.cmf3(rgb_f[2], d_f[2], f_4)
        f_2 = self.cmf2(rgb_f[1], d_f[1], f_3)
        f_1 = self.cmf1(rgb_f[0], d_f[0], f_2)

        out_fus1234, out_fus234, out_fus34 = self.R_CLCM_1234(f_1, f_2, f_3, f_4)
        out_fus4321, out_fus321, out_fus21 = self.R_CLCM_4321(f_4, f_3, f_2, f_1)

        fus_3421 = self.conv_3(torch.cat((out_fus34 + out_fus321, out_fus34 * out_fus321), 1))
        fus_2341 = self.conv_2(torch.cat((out_fus21 + out_fus234, out_fus21 * out_fus234), 1))

        return out_fus1234, fus_2341, fus_3421, out_fus4321, f_g


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
        self.rgb_net = model.encoder
        self.t_net = model.encoder
        self.R_CLCM_1 = R_CLCM(64, 128, 320, 512)
        self.R_CLCM_2 = R_CLCM(512, 320, 128, 64)
        self.conv_3 = convblock(320 * 2, 320, 3, 1, 1)
        self.conv_2 = convblock(128 * 2, 128, 3, 1, 1)
        self.cmf1 = CMF(64, 128)
        self.cmf2 = CMF(128, 320)
        self.cmf3 = CMF(320, 512)
        self.cmf4 = CMF(512, 512)
        self.decoder = Decoder()
        self.sig = nn.Sigmoid()

        self.up_g = convblock(512, 64, 3, 1, 1)
        self.up_4 = convblock(512, 64, 3, 1, 1)
        self.up_3 = convblock(320, 64, 3, 1, 1)
        self.up_2 = convblock(128, 64, 3, 1, 1)
        self.up_1 = convblock(64, 64, 1, 1, 0)

        self.up_4_ = convblock(512, 64, 3, 1, 1)
        self.up_3_ = convblock(320, 64, 3, 1, 1)
        self.up_2_ = convblock(128, 64, 3, 1, 1)
        self.up_1_ = convblock(64, 64, 1, 1, 0)

        self.conv_s1 = convblock(64 * 5, 64, 3, 1, 1)
        self.score1 = nn.Conv2d(64, 1, 1, 1, 0)

        self.conv_s2 = convblock(64 * 5, 64, 3, 1, 1)
        self.score2 = nn.Conv2d(64, 1, 1, 1, 0)

        self.score = nn.Conv2d(2, 1, 1, 1, 0)

    def forward(self, imgs, depths):
        img_1, img_2, img_3, img_4 = self.rgb_net(imgs)
        dep_1, dep_2, dep_3, dep_4 = self.t_net(depths)

        out_r1234, out_r234, out_r34 = self.R_CLCM_1(img_1, img_2, img_3, img_4)
        out_r4321, out_r321, out_r21 = self.R_CLCM_2(img_4, img_3, img_2, img_1)
        out_r3421 = self.conv_3(torch.cat((out_r321 + out_r34, out_r321 * out_r34), 1))
        out_r2341 = self.conv_2(torch.cat((out_r21 + out_r234, out_r21 * out_r234), 1))

        out_d1234, out_d234, out_d34 = self.R_CLCM_1(dep_1, dep_2, dep_3, dep_4)
        out_d4321, out_d321, out_d21 = self.R_CLCM_2(dep_4, dep_3, dep_2, dep_1)
        out_d3421 = self.conv_3(torch.cat((out_d321 + out_d34, out_d321 * out_d34), 1))
        out_d2341 = self.conv_2(torch.cat((out_d21 + out_d234, out_d21 * out_d234), 1))

        r_f_list = [img_1, img_2, img_3, img_4]
        d_f_list = [dep_1, dep_2, dep_3, dep_4]
        cmcl_fus1234, cmcl_fus2341, cmcl_fus3421, cmcl_fus4321, f_g = self.decoder(r_f_list, d_f_list)

        clcm_fus4321 = self.cmf4(out_r4321, out_d4321, f_g)
        clcm_fus3421 = self.cmf3(out_r3421, out_d3421, clcm_fus4321)
        clcm_fus2341 = self.cmf2(out_r2341, out_d2341, clcm_fus3421)
        clcm_fus1234 = self.cmf1(out_r1234, out_d1234, clcm_fus2341)

        up_g = self.up_g(F.interpolate(f_g, clcm_fus1234.size()[2:], mode='bilinear', align_corners=True))

        clcm_4 = self.up_4(F.interpolate(clcm_fus4321, clcm_fus1234.size()[2:], mode='bilinear', align_corners=True))
        clcm_3 = self.up_3(F.interpolate(clcm_fus3421, clcm_fus1234.size()[2:], mode='bilinear', align_corners=True))
        clcm_2 = self.up_2(F.interpolate(clcm_fus2341, clcm_fus1234.size()[2:], mode='bilinear', align_corners=True))
        clcm_1 = self.up_1(clcm_fus1234)

        s_clcm = self.conv_s1(torch.cat((clcm_1, clcm_2, clcm_3, clcm_4, up_g), 1))
        score_clcm = self.score1(F.interpolate(s_clcm, (384, 384), mode='bilinear', align_corners=True))

        cmcl_4 = self.up_4_(F.interpolate(cmcl_fus4321, cmcl_fus1234.size()[2:], mode='bilinear', align_corners=True))
        cmcl_3 = self.up_3_(F.interpolate(cmcl_fus3421, cmcl_fus1234.size()[2:], mode='bilinear', align_corners=True))
        cmcl_2 = self.up_2_(F.interpolate(cmcl_fus2341, cmcl_fus1234.size()[2:], mode='bilinear', align_corners=True))
        cmcl_1 = self.up_1_(cmcl_fus1234)

        s_cmcl = self.conv_s2(torch.cat((cmcl_1, cmcl_2, cmcl_3, cmcl_4, up_g), 1))
        score_cmcl = self.score2(F.interpolate(s_cmcl, (384, 384), mode='bilinear', align_corners=True))

        score = self.score(torch.cat((score_clcm + torch.mul(score_clcm, self.sig(score_cmcl)),
                                      score_cmcl + torch.mul(score_cmcl, self.sig(score_clcm))), 1))

        return score, score_clcm, score_cmcl, self.sig(score)

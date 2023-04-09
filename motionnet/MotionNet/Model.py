# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import torch.nn.functional as F
import torch.nn as nn
import torch


class CellClassification(nn.Module):
    def __init__(self, category_num=5):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(64)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x

def TemporalPooling(x, batch):
    x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = F.adaptive_avg_pool3d(x, (1, None, None)) #F.adaptive_max_pool3d(x, (1, None, None))
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()
    return x

class STPN(nn.Module):
    def __init__(self, height_feat_size=13, T=5):
        super(STPN, self).__init__()

        self.T = T

        # Inital, change to conv3d to get a more complete map
        self.conv_pre_1 = Conv3D(height_feat_size, 64, kernel_size=(3,3,3), stride=1, padding=1)
        if self.T >= 5:
            self.conv_pre_2 = Conv3D(64, 64, kernel_size=(3,3,3), stride=1, padding=1)
        if self.T >= 7:
            self.conv_pre_3 = Conv3D(64, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1))

        # self.conv_pre_1 = nn.Conv2d(height_feat_size, 64, kernel_size=3, stride=1, padding=1)
        # self.conv_pre_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn_pre_1 = nn.BatchNorm2d(64)
        # self.bn_pre_2 = nn.BatchNorm2d(64)

        
        # Account for different time sizes

        # self.conv3d_1 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_2 = Conv3D(256, 256, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_3 = Conv3D(512, 512, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv3d_4 = Conv3D(1024, 1024, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        # You will have to change the kernel size here depending on your T (past frames)
        if self.T >= 3:
            self.conv3d_1 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        if self.T >= 5:
            self.conv3d_2 = Conv3D(256, 256, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        if self.T >= 7:
            self.conv3d_3 = Conv3D(512, 512, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        if self.T >= 11:
            self.conv3d_4 = Conv3D(1024, 1024, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))


        self.conv1_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(1024 + 512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)


        self.bn1_1 = nn.BatchNorm2d(128)
        self.bn1_2 = nn.BatchNorm2d(128)

        self.bn2_1 = nn.BatchNorm2d(256)
        self.bn2_2 = nn.BatchNorm2d(256)

        self.bn3_1 = nn.BatchNorm2d(512)
        self.bn3_2 = nn.BatchNorm2d(512)

        self.bn4_1 = nn.BatchNorm2d(1024)
        self.bn4_2 = nn.BatchNorm2d(1024)

        self.bn5_1 = nn.BatchNorm2d(512)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.bn6_1 = nn.BatchNorm2d(256)
        self.bn6_2 = nn.BatchNorm2d(256)

        self.bn7_1 = nn.BatchNorm2d(128)
        self.bn7_2 = nn.BatchNorm2d(128)

        self.bn8_1 = nn.BatchNorm2d(64)
        self.bn8_2 = nn.BatchNorm2d(64)


    def forward(self, x):
        batch, seq, z, h, w = x.size()
        # x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        # x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        # x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
        x = self.conv_pre_1(x)
        if self.T >= 5:
            x = self.conv_pre_2(x)
        if self.T >= 7:
            x = self.conv_pre_3(x)
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()
        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        if self.T >= 3:
            x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_1 = self.conv3d_1(x_1)
            x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        if self.T >= 5:
            x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_2 = self.conv3d_2(x_2)
            x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        if self.T >= 7:
            x_3 = x_3.view(batch, -1, x_3.size(1), x_3.size(2), x_3.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_3 = self.conv3d_3(x_3)
            x_3 = x_3.view(-1, x_3.size(2), x_3.size(3), x_3.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1
        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        if self.T >= 11:
            x_4 = x_4.view(batch, -1, x_4.size(1), x_4.size(2), x_4.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_4 = self.conv3d_4(x_4)
            x_4 = x_4.view(-1, x_4.size(2), x_4.size(3), x_4.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -------------------------------- Temporal Pooling Path --------------------------------
        x = TemporalPooling(x, batch)
        x_1 = TemporalPooling(x_1, batch)
        x_2 = TemporalPooling(x_2, batch)
        x_3 = TemporalPooling(x_3, batch)
        x_4 = TemporalPooling(x_4, batch)


        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))

        # res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))
        res_x = F.relu(self.bn8_2(self.conv8_2(F.interpolate(x_8, scale_factor=(2, 2)))))

        return res_x


class MotionNet(nn.Module):
    def __init__(self, num_classes=11, height_feat_size=13, T=5):
        super(MotionNet, self).__init__()

        self.cell_classify = CellClassification(category_num=num_classes)
        self.stpn = STPN(height_feat_size=height_feat_size, T=T)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs)
        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        return cell_class_pred


# For MGDA loss computation
class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class MotionNetMGDA(nn.Module):
    def __init__(self, classes=11):
        super(MotionNetMGDA, self).__init__()

        self.cell_classify = CellClassification(category_num=classes)

    def forward(self, stpn_out):
        # Cell Classification head
        cell_class_pred = self.cell_classify(stpn_out)

        return cell_class_pred
import torch
import torch.nn as nn

from .pointPWC import PointConvD, PointConvUp


class ScoreNet(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16):
        super(ScoreNet, self).__init__()
        self.nsample = nsample
        self.level1 = PointConvD(2048, self.nsample, in_channel+3, out_channel)
        self.level2 = PointConvD(512, self.nsample, out_channel+3, out_channel)
        self.level3 = PointConvD(256, self.nsample, out_channel+3, out_channel)
        self.uplevel3 = PointConvUp(16, out_channel + out_channel + 3, [64, 64])
        self.uplevel2 = PointConvUp(16, out_channel + 64 + 3, [64, 64])
        self.uplevel1 = PointConvUp(16, 3 + 64 + 3, [64, 64])

        self.conv1d1 = nn.Conv1d(64, 32, 1)
        self.conv1d2 = nn.Conv1d(32, 1, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, xyz, flow):
        
        xyz1, feat1, _ = self.level1(xyz, flow)
        xyz2, feat2, _ = self.level2(xyz1, feat1)
        xyz3, feat3, _ = self.level3(xyz2, feat2)
        # print(feat3.shape)
        up_feat3 = self.uplevel3(xyz2, xyz3, feat2, feat3)
        up_feat2 = self.uplevel2(xyz1, xyz2, feat1, up_feat3)
        up_feat1 = self.uplevel1(xyz, xyz1, xyz, up_feat2)
        
        x = self.conv1d1(up_feat1)
        x = self.bn1(x)
        x = self.conv1d2(x)
        x = torch.sigmoid(x).squeeze(1)

        return x
from operator import index
from turtle import color
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.pointConv import PointConv, PointConvD

from .net_utils import *

LEAKY_RATE = 0.1
use_bn = False
scale = 1.0


class getCostVolume(nn.Module):
    """Calculate Cost Volume.

    Args:
        init: nsample, in_channel, mlp, bn, use_leaky
        forward:xyz1, xyz2, points1, points2
    """
    def __init__(self, nsample, in_channel, mlp, bn=use_bn, use_leaky=True):
        super(getCostVolume, self).__init__()
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.bn = bn
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        B, C, N1 = xyz1.shape
        _, D1, _ = points1.shape

        # [B, N, C]
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1)           # [B, N1, nsample]
        neighbor_xyz = index_points_group(xyz2, knn_idx)        # [B, N1, nsample, 3]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)   # [B, N1, nsample, 3]

        grouped_points2 = index_points_group(points2, knn_idx)                      # [B, N1, nsample, D2]
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)  # [B, N1, nsample, D1]

        cost_volumes_init = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1) # [B, N1, nsample, D1+D2+3]
        cost_volumes_init = cost_volumes_init.permute(0, 3, 2, 1)                                # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            cost_volumes_init = conv(cost_volumes_init)
            if self.bn:
                cost_volumes_init = self.mlp_bns[i](cost_volumes_init)
            cost_volumes_init = self.relu(cost_volumes_init)
        
        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))    # [B, C, nsample, N1]
        point2patch_cost = torch.sum(weights * cost_volumes_init, dim=2)     # [B, C, N]

        # patch-to-patch volume
        knn_idx = knn_point(self.nsample, xyz1, xyz1)           # [B, N1, nsample]
        neighbor_xyz = index_points_group(xyz1, knn_idx)        # [B, N1, nsample, 3]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)   # [B, N1, nsample, 3]

        grouped_point2patch_cost = index_points_group(point2patch_cost.permute(0, 2, 1), knn_idx)       # [B, N1, nsample, C]

        # weighted sum
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))    # [B, C, nsample, N1]
        patch2patch_cost = torch.sum(weights * grouped_point2patch_cost.permute(0, 3, 2, 1), dim=2)     # [B, C, N]

        return patch2patch_cost


def warp(xyz1, xyz2, flow=None):
    """get the warpped points. P' = P + flow

    Args:
        xyz1 : init points position
        xyz2 : goal points position
        flow : predict scene flow

    Returns:
        warped_xyz2
    """
    if flow is None:
        return xyz2
    
    xyz1_to_xyz2 = xyz1 + flow      # [B, N1, 3]

    # interpolate flow
    B, C, _ = xyz1.shape
    _, _, N2 = xyz2.shape
    xyz1_to_xyz2 = xyz1_to_xyz2.permute(0, 2, 1)    # [B, 3, N1]
    xyz2 = xyz2.permute(0, 2, 1)                    # [B, 3, N2]
    flow = flow.permute(0, 2, 1)

    knn_idx = knn_point(3, xyz1_to_xyz2, xyz2)      # [B, N2, 3]
    grouped_xyz_norm = index_points_group(xyz1_to_xyz2, knn_idx) - xyz2.view(B, N2, 1, C)   # [B, N2, 3, C]
    dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
    norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
    weight = (1.0 / dist) / norm

    # generate reverse flow
    grouped_flow = index_points_group(flow, knn_idx)
    flow_reverse = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow, dim=2)

    # warp
    warped_xyz2 = (xyz2 - flow_reverse).permute(0, 2, 1)

    return warped_xyz2


def Upsample(xyz, sparse_xyz, sparse_flow):
    B, C, N = xyz.shape

    xyz = xyz.permute(0, 2, 1)
    sparse_xyz = sparse_xyz.permute(0, 2, 1)
    sparse_flow = sparse_flow.permute(0, 2, 1)

    # interpolate flow
    knn_idx = knn_point(3, sparse_xyz, xyz)
    grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
    dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
    norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
    weight = (1.0 / dist) / norm
    
    # generate flow using sparse flow
    grouped_flow = index_points_group(sparse_flow, knn_idx)
    dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(0, 2, 1)

    return dense_flow


class SceneFlowEstimator(nn.Module):
    def __init__(self, feat_c, cost_c, flow_c=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-200, 200], use_leaky=True):
        super(SceneFlowEstimator, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        last_channel = feat_c + cost_c + flow_c

        self.pointconv_list = nn.ModuleList()
        for _, ch_out in enumerate(channels):
            self.pointconv_list.append(PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True))
            last_channel = ch_out
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out
        
        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points =torch.cata([feats, cost_volume, flow], dim=1)
        
        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)
        
        flow = self.fc(new_points)
        return new_points, flow.clap(self.clamp[0], self.clamp[1])


class PointPWC(nn.Module):
    def __init__(self):
        super(PointPWC, self).__init__()

        flow_nsample = 32
        feat_nsample = 16
        self.scale = scale

        # l0: 8192
        self.level0_0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        self.cost0 = getCostVolume(flow_nsample, 32+32+32+32+3, [32, 32])
        self.flow0 = SceneFlowEstimator(32+64, 32)
        self.level0_2 = Conv1d(32, 64)

        # l1: 2048
        self.level1 = PointConvD(2048, feat_nsample, 64+3, 64)
        self.cost1 = getCostVolume(flow_nsample, 64+32+64+32+3, [64, 64])
        self.flow1 = SceneFlowEstimator(64+64, 64)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        # l2: 512
        self.level2 = PointConvD(512, feat_nsample, 128+3, 128)
        self.cost2 = getCostVolume(flow_nsample, 128+64+128+64+2, [128, 128])
        self.flow2 = SceneFlowEstimator(128+64, 128)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        # l3: 256
        self.level3 = PointConvD(256, feat_nsample, 256+3, 256)
        self.cost3 = getCostVolume(flow_nsample, 256+64+256+64+3, [256, 256])
        self.flow3 = SceneFlowEstimator(256, 256, flow_c=0)                     # have no init flow
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        # l4: 64
        self.level4 = PointConvD(64, feat_nsample, 512+3, 256)

        # deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)


    def forward(self, xyz1, xyz2, color1, color2):
        pc1 = xyz1.permute(0, 2, 1)
        pc2 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)
        color2 = color2.permute(0, 2, 1)

        # l0
        feat1_l0 = self.level0_0(color1)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        feat2_l0 = self.level0_0(color2)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

        # feature extraction.
        # l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1, feat1_l0_1)
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2, feat2_l0_1)
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)

        # l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l0_1)
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)

        # l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3_4 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3_4)

        # l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = Upsample(pc1_l3, pc1_l4, feat1_l4)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = Upsample(pc2_l3, pc2_l4, feat2_l4)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        # calculate flow & upsample
        # l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim=1)

        cost3 = self.cost3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cost3)
        
        feat1_l3_2 = Upsample(pc1_l2, pc1_l3, feat1_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)
        feat2_l3_2 = Upsample(pc2_l2, pc2_l3, feat2_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)
        feat3_up = Upsample(pc1_l2, pc1_l3, feat3)

        # l2
        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim=1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim=1)

        up_flow2 = Upsample(pc1_l2, pc1_l3, self.scale*flow3)
        pc2_l2_warp = warp(pc1_l2, pc2_l2, up_flow2)
        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat2_l2)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim=1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cost2, up_flow2)

        feat1_l2_1 = Upsample(pc1_l1, pc1_l2, feat1_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)
        feat2_l2_1 = Upsample(pc2_l1, pc2_l2, feat2_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)
        feat2_up = Upsample(pc1_l1, pc1_l2, feat2)

        # l1
        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        up_flow1 = Upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = warp(pc1_l1, pc2_l1, up_flow1)
        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cost1, up_flow1)

        feat1_l1_0 = self.upsample(pc1, pc1_l1, feat1_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)
        feat2_l1_0 = self.upsample(pc2, pc2_l1, feat2_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)
        feat1_up = self.upsample(pc1, pc1_l1, feat1)

        # l1
        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)

        up_flow0 = self.upsample(pc1, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1, pc2, up_flow0)
        cost0 = self.cost0(pc1, pc2_l0_warp, c_feat1_l0, c_feat2_l0)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1, new_feat1_l0, cost0, up_flow0)

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2
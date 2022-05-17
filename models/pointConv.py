import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils
from .net_utils import *

LEAKY_RATE = 0.1
use_bn = False
        

class PointConv(nn.Module):
    """
    Args:
        in_channel, out_channel,hidden_unit, weightnet, bn, use_leaky
    """
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv, self).__init__()
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)
        else:
            self.bn_linear = None
        
        self.relu = nn.ReLu(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """PointConv without strides size, i.e., the input and output have the same number of points.

        Args:
            xyz ([B, C, N]): input points position data
            points ([B, D, N]): input points data
        Return:
            new_xyz: sampled points position data
            new_points_concat: sample points feature data
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, N, -1)
        new_points = self.linear(new_points)
        if self.bn_linear is not None:
            new_points = self.bn_linear(new_points(0, 2, 1))
        else:
            new_points =  new_points.permute(0, 2, 1)
        
        new_points = self.relu(new_points)

        return new_points


class PointConvD(nn.Module):
    """PointConv with downsampling.

    Args:
        in_channel, out_channel,hidden_unit, weightnet, bn, use_leaky
    """
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)
        
        self.relu = nn.ReLu(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        B = xyz.shape[0]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_group(xyz, fps_idx)
        
        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn_linear is not None:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)
        
        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils


LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.ReLu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None
    
    def forward(self, x):
        x = self.conv1(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.ReLu(x)

        return x
    

def square_distance(src, dst):
    """Calculate Euclid distance between each two points.

    Args:
        src ([B, N, C]): source points
        dst ([B, N, C]): target points
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    # -2 * src^T  * dst
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # x_n * x_n + y_n * y_n + z_n * z_n
    dist += torch.sum(src**2, -1).view(B, N, 1)
    # x_m * x_m + y_m * y_m + z_m * z_m
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    return dist

def knn_point(nsample, xyz, new_xyz):
    """get new points feature using nsample.

    Args:
        nsample: max sample number in local region
        xyz ([B, N, C]): all (origion) points
        new_xyz ([B, S, C]): query points
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)

    return group_idx


def index_points_group(points, knn_idx):
    """Using knn_idx to find the corresponding points in input points.
    Args:
        points ([B, N, C]): input points data
        knn_index ([B, N, K]): sample index data

    Returns:
        new_points: indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)
    return new_points


def group(nsample, xyz, points):
    """Sample & group n points from xyz.

    Args:
        nsample (_type_): scalar
        xyz ([B, N, C]): input points position data
        points ([B, N, D]): input points data
    Return:
        new_points([B, 1, N, C+D]): sampled points data
        new_xyz([B, 1, C]): sampled points position data
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    return new_points, grouped_xyz_norm


def group_query(nsample, s_xyz, xyz, s_points):
    """

    Args:
        s_xyz ([B, N, C]): input points position data
        xyz ([B, S, C]): input points position data (p2)
        s_points ([B, N, D]): input points data
    """
    B, _, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    return new_points, grouped_xyz_norm


class WeightNet(nn.Module):
    """MLP

    Args:
        in_channel, out_channel,hidden_unit, bn
    """
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[i-1], hidden_unit[i], 1))
                self.mlp_bn.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        x = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            x = conv(x)
            if self.bn:
                x = self.mlp_bns[i](x)
            x = F.relu(x)
        
        return x
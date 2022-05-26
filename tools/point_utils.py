import os
import torch
import numpy as np

from pointnet2 import pointnet2_utils


LEAKY_RATE = 0.1
use_bn = False

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


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()


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


def get_batch_2d_flow(pc1, pc2, predicted_pc2, paths):
    if 'KITTI' in paths[0] or 'kitti' in paths[0]:
        focallengths = []
        cxs = []
        cys = []
        constx = []
        consty = []
        constz = []
        for path in paths:
            fname = os.path.split(path)[-1]
            calib_path = os.path.join(
                os.path.dirname(__file__),
                'calib_cam_to_cam',
                fname + '.txt')
            with open(calib_path) as fd:
                lines = fd.readlines()
                P_rect_left = \
                    np.array([float(item) for item in
                              [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                             dtype=np.float32).reshape(3, 4)
                focallengths.append(-P_rect_left[0, 0])
                cxs.append(P_rect_left[0, 2])
                cys.append(P_rect_left[1, 2])
                constx.append(P_rect_left[0, 3])
                consty.append(P_rect_left[1, 3])
                constz.append(P_rect_left[2, 3])
        focallengths = np.array(focallengths)[:, None, None]
        cxs = np.array(cxs)[:, None, None]
        cys = np.array(cys)[:, None, None]
        constx = np.array(constx)[:, None, None]
        consty = np.array(consty)[:, None, None]
        constz = np.array(constz)[:, None, None]

        px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                          constx=constx, consty=consty, constz=constz)
    else:
        px1, py1 = project_3d_to_2d(pc1)
        px2, py2 = project_3d_to_2d(predicted_pc2)
        px2_gt, py2_gt = project_3d_to_2d(pc2)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y
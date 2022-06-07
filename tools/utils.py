import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sigmoid_rampup(current, rampup_length, s):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242 . 
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        # phase = current / rampup_length
        return float(np.exp(-s * phase * phase))


def DCGTGenerator(l_pred, r_pred, l_flawmap, r_flawmap):
        _, _, C = l_pred.shape
        l_mask = (r_flawmap >= l_flawmap).float()
        r_mask = (l_flawmap >= r_flawmap).float()
        l_mask = l_mask.unsqueeze(2).repeat(1,1,C)
        r_mask = r_mask.unsqueeze(2).repeat(1,1,C)
        l_dc_gt = l_mask * l_pred + (1 - l_mask) * r_pred
        r_dc_gt = r_mask * r_pred + (1 - r_mask) * l_pred
        return l_dc_gt, r_dc_gt


def MNGTGenerator(pos1, pred, gt):
        diff1 = torch.abs_(gt - pred.permute(0, 2, 1))
        diff2 = torch.sum(diff1, dim=2, keepdim=True)
        # diff2 = self.blur(pos1, diff2, self.k)
        # dmax = diff2.max(dim=1, keepdim=True)[0]
        # dmin = diff2.min(dim=1, keepdim=True)[0]
        # diff2.sub_(dmin).div_(dmax - dmin + 1e-9)
        # x = torch.sigmoid(diff2)
        diff2 =  1.0 - torch.exp(-10.0 * diff2)

        mn_gt = diff2.squeeze(2)
        return mn_gt
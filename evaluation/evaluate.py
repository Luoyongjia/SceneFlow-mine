"""
load model + calculate EPE3D, ACCS, ACCR, Outliers, EPE2D, ACC2D
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from loss.supervised_loss import multiScaleLoss

from tools import *
from tqdm import tqdm
from models import get_model
from indicator import *


def eval(val_loader, args, checkpoint_dir):
    # log init
    logger = logging.getLogger(args.exp_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(args.log_dir) + 'train_%s_sceneflow.txt'%args.exp_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # state init
    epe3ds = AverageMeter()
    acc3d_s = AverageMeter()
    acc3d_r = AverageMeter()
    outliers = AverageMeter()

    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    time_sum = AverageMeter()

    total_loss = 0
    total_seen = 0

    # load model
    pretrain = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    model = get_model(args.model.name)
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    for i in args.eval.epoch:
        start_time = time.time()
        idx = 0
        for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
            model = model.eval()
            pos1, pos2, norm1, norm2, flow, path = data

            # move to cuda
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda()

            with torch.no_grad():
                pred_flows, fps_pc1_idxs, _, pc1, pc2 = model(pos1, pos2, norm1, norm2)

                loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

                full_flow = pred_flows[0].permute(0, 2, 1)
            
            end_time = time.time()
            time_sum.update(end_time - start_time)

        total_loss += loss.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = pos1.cpu().numpy()
        pc2_np = pos2.cpu().numpy() 
        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_s, acc3d_r, outlier = evaluate_3D(pred_sf, sf_np)
        epe3ds.update(EPE3D)
        acc3d_s.update(acc3d_s)
        acc3d_r.update(acc3d_r)
        outliers.update(outlier)

        flow_pred, flow_gt = get_batch_2d_flow(pc1_np, pc1_np+sf_np, pc1_np+pred_sf, path)
        EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)
        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

        mean_loss = total_loss / total_seen

        print(f"Evaluate mean loss: {mean_loss}, time: {time_sum.avg}")
        logger.info(f"Evaluate mean loss: {mean_loss},  time: {time_sum.avg}")

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
            'ACC3DS {acc3d_s.avg:.4f}\t'
            'ACC3DR {acc3d_r.avg:.4f}\t'
            'Outliers3D {outlier_.avg:.4f}\t'
            'EPE2D {epe2d_.avg:.4f}\t'
            'ACC2D {acc2d_.avg:.4f}\t'
            'time {time_sum.avg:.4f}'
            .format(
                    epe3d_=epe3ds,
                    acc3d_s=acc3d_s,
                    acc3d_r=acc3d_r,
                    outlier_=outliers,
                    epe2d_=epe2ds,
                    acc2d_=acc2ds,
                    time_sum=time_sum,
                    ))

    print(res_str)
    logger.info(res_str)

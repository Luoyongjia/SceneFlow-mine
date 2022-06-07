import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict

from evaluation import *
from models import get_model
from loss import *
from tools import *


blue = lambda x: '\033[94m' + x + '\033[0m'
def train_dual(labeled_train_loader, unlabeled_train_loader, val_loader, logger, board, args):
    """trainer of semi points sceneflow predictor network.

    Args:
        train_loader:train dataset
        val_loader: val dataset
        logger
        board: tensorboard
        args: _description_
    """

    # init model
    model_l, model_r, model_mn = get_model(args.model.name)
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        model_l.cuda(device_ids[0])
        model_r.cuda(device_ids[0])
        model_mn.cuda(device_ids[0])
        model_l = torch.nn.DataParallel(model_l, device_ids = device_ids)
        model_r = torch.nn.DataParallel(model_r, device_ids = device_ids)
        model_mn = torch.nn.DataParallel(model_mn, device_ids = device_ids)
    else:
        model_l.cuda()
        model_r.cuda()
        model_mn.cuda()
    
    checkpoints_dir = os.path.join(args.log_dir, args.name, '/checkpoint/')
    
    # init optimizer & scheduler
    if args.optimizer == 'SGD':
        optimizer_l = torch.optim.SGD(model_l.parameters(), lr=args.learning_rate, momentum=0.9)
        optimizer_r = torch.optim.SGD(model_r.parameters(), lr=args.learning_rate, momentum=0.9)
        optimizer_mn = torch.optim.SGD(model_mn.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_l = torch.optim.Adam(model_l.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
        optimizer_r = torch.optim.Adam(model_r.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
        optimizer_mn = torch.optim.Adam(model_mn.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)  

    optimizer_l.param_groups[0]['initial_lr'] = args.learning_rate 
    optimizer_r.param_groups[0]['initial_lr'] = args.learning_rate 
    optimizer_mn.param_groups[0]['initial_lr'] = args.learning_rate 
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    scheduler_r = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    scheduler_mn = torch.optim.lr_scheduler.StepLR(optimizer_mn, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5 

    # indicator init
    best_epe = 1000.0
    labeled_batch_size = args.train.labeled_batch_size
    unlabeled_batch_size = args.train.unlabeled_batch_size

    # pretrain = args.pretrain 
    # init_epoch = args.pretrain.epochs if args.pretrain is not None else 0 
    init_epoch = 0
    
    for epoch in range(init_epoch, args.train.epoch):
        lr = max(optimizer_l.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer_l.param_groups:
            param_group['lr'] = lr
        
        total_sup_loss = 0
        total_unsup_loss = 0
        total_loss = 0
        total_seen = 0
        total_mn_loss = 0

        optimizer_l.zero_grad()
        optimizer_r.zero_grad()

        unlabeled_iter = enumerate(unlabeled_train_loader)

        for i, data in tqdm(enumerate(labeled_train_loader, 0), total=len(labeled_train_loader), smoothing=0.9):
            try:
                _, unlabeled_data = next(unlabeled_iter)
            except StopIteration:
                trainloader_unlabeled_iter = enumerate(unlabeled_train_loader)
                _, unlabeled_data = next(unlabeled_iter)
            
            labeled_pos1, labeled_pos2, labeled_norm1, labeled_norm2, labeled_flow, _ = data  
            unlabeled_pos1, unlabeled_pos2, unlabeled_norm1, unlabeled_norm2, unlabeled_flow, _ = unlabeled_data  
            
            # move to cuda 
            cur_steps = len(labeled_train_loader) * epoch + i
            total_steps = len(labeled_train_loader) * args.epochs
            dc_scale = sigmoid_rampup(cur_steps, total_steps, 5.0)
            # dc_scale = 0

            pos1 = torch.cat((labeled_pos1.cuda(), unlabeled_pos1.cuda()), 0)
            pos2 = torch.cat((labeled_pos2.cuda(), unlabeled_pos2.cuda()), 0)
            norm1 = torch.cat((labeled_norm1.cuda(), unlabeled_norm1.cuda()), 0)
            norm2 = torch.cat((labeled_norm2.cuda(), unlabeled_norm2.cuda()), 0)
            labeled_flow = labeled_flow.cuda()

            model_l = model_l.train()
            model_r = model_r.train()
            model_mn = model_mn.train()

            # get mutual supervision labels
            with torch.no_grad():
                pred_flows_l, _, _, pc1_l, pc2_l = model_l(pos1, pos2, norm1, norm2)
                pred_flows_r, _, _, pc1_r, pc2_r = model_r(pos1, pos2, norm1, norm2)
            mask_pred_l = model_mn(pc1_l[0], pred_flows_l[0])
            mask_pred_r = model_mn(pc1_r[0], pred_flows_r[0])
            with torch.no_grad():
                dc_gt_l, dc_gt_r = DCGTGenerator(pred_flows_l[0].permute(0, 2, 1), pred_flows_r[0].permute(0, 2, 1), mask_pred_l.detach(), mask_pred_r.detach())

            # -------------------------------------------------------------------------------------------------------
            # train Scene Flow networks
            # fix the mn's parameters
            for index, (name, value) in enumerate(model_mn.named_parameters()):
                value.requires_grad = False

            # l
            pred_flows_l, fps_pc1_idxs_l, _, pc1_l, pc2_l = model_l(pos1, pos2, norm1, norm2)
            pred_flows_r, fps_pc1_idxs_r, _, pc1_r, pc2_r = model_r(pos1, pos2, norm1, norm2)
            
            # dividing to labeled and unlabeled part
            labeled_flows_l = []
            labeled_flows_r = []
            labeled_fps_pc1_idxs_l = []
            labeled_fps_pc1_idxs_r = []

            for i in range(len(pred_flows_l)):
                labeled_flow_l = pred_flows_l[i][0: args.train.labeled_batch_size]
                labeled_flow_r = pred_flows_r[i][0: args.train.labeled_batch_size]
                labeled_flows_l.append(labeled_flow_l)
                labeled_flows_r.append(labeled_flow_r)

            for i in range(len(fps_pc1_idxs_l)):
                labeled_fps_pc1_idx_l = fps_pc1_idxs_l[i][0:args.train.labeled_batch_size,...]
                labeled_fps_pc1_idx_r = fps_pc1_idxs_r[i][0:args.train.labeled_batch_size,...]
                labeled_fps_pc1_idxs_l.append(labeled_fps_pc1_idx_l)
                labeled_fps_pc1_idxs_r.append(labeled_fps_pc1_idx_r)
            
            sup_loss_l = multiScaleLoss(labeled_flows_l, labeled_flow_l, labeled_fps_pc1_idxs_l)
            sup_loss_r = multiScaleLoss(labeled_flows_r, labeled_flow_r, labeled_fps_pc1_idxs_r)

            unsup_loss_l = multiScaleLoss(pred_flows_l, dc_gt_l, fps_pc1_idxs_l)
            unsup_loss_r = multiScaleLoss(pred_flows_r, dc_gt_r, fps_pc1_idxs_r)

            loss_l = sup_loss_l + dc_scale * unsup_loss_l
            loss_r = sup_loss_r + dc_scale * unsup_loss_r

            loss_l.backward()
            optimizer_l.step()
            optimizer_l.zero_grad()

            loss_r.backward()
            optimizer_r.step()
            optimizer_r.zero_grad()

            total_sup_loss += sup_loss_l.item() * args.train.labeled_batch_size
            total_unsup_loss += unsup_loss_l.item() * args.train.labeled_batch_size
            total_loss += loss_l.cpu().data * args.train.labeled_batch_size
            total_seen += args.train.labeled_batch_size

            # -------------------------------------------------------------------------------------------------------
            # train mask network
            for index, (name, value) in enumerate(model_mn.named_parameters()):
                value.requires_grad = True

            with torch.no_grad():
                labeled_flow_l = pred_flows_l[0][0:args.train.labeled_batch_size,...]
                labeled_flow_r = pred_flows_r[0][0:args.train.labeled_batch_size,...]
                mn_gt_l = MNGTGenerator(labeled_pos1, labeled_flow_l.detach(), labeled_flow)
                mn_gt_r = MNGTGenerator(labeled_pos1, labeled_flow_r.detach(), labeled_flow)
            
            mask_pred_l_labeled = mask_pred_l[0:args.train.labeled_batch_size,...]
            mask_pred_r_labeled = mask_pred_r[0:args.train.labeled_batch_size,...]

            mn_loss_l = torch.mean(F.mse_loss(mask_pred_l_labeled, mn_gt_l, reduction='none'), dim=(0,1))
            mn_loss_r = torch.mean(F.mse_loss(mask_pred_r_labeled, mn_gt_r, reduction='none'), dim=(0,1))

            optimizer_mn.zero_grad()
            loss_mn = mn_loss_l + mn_loss_r
            loss_mn.backward()
            optimizer_mn.step()  
            total_mn_loss += loss_mn.cpu().data * args.train.labeled_batch_size

        scheduler_l.step()
        scheduler_r.step()
        scheduler_mn.step()

        sup_loss = total_sup_loss/total_seen
        unsup_loss = total_unsup_loss/total_seen
        train_loss = total_loss / total_seen
        mn_loss = total_mn_loss/total_seen
        
        unsup_loss = total_unsup_loss/total_seen
        mn_loss = total_mn_loss/total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        print(str_out)
        logger.info(str_out)
        
        str_out1 = 'EPOCH %d %s mean sup_loss: %f'%(epoch, blue('train'), sup_loss)
        str_out2 = 'EPOCH %d %s mean unsup_loss: %f'%(epoch, blue('train'), unsup_loss)
        str_out3 = 'EPOCH %d %s mean mn_loss: %f'%(epoch, blue('train'), mn_loss)
        print(str_out1)
        print(str_out2)
        print(str_out3)
        board.add_scalar("sup_loss", sup_loss, epoch)
        board.add_scalar("unsup_loss", unsup_loss, epoch)
        board.add_scalar("mn_loss", mn_loss, epoch)
        # print(str_out4)
        logger.info(str_out1)
        logger.info(str_out2)
        logger.info(str_out3)
        # logger.info(str_out4)

        # eval_epe3d, eval_loss, eval_chamfer_loss, eval_smoothness_loss, eval_curvature_loss = eval_sceneflow(model_l.eval(), val_loader)
        eval_epe3d, sup_loss = eval_sceneflow(model_l.eval(), val_loader)
        str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, sup_loss)
        print(str_out)
        logger.info(str_out)
        board.add_scalar("epe3d", eval_epe3d, epoch)

        if eval_epe3d < best_epe:
            best_epe = eval_epe3d
            torch.save(optimizer_l.state_dict(), '%s/optimizer.pth'%(checkpoints_dir))
            if args.multi_gpu is not None:
                torch.save(model_l.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            else:
                torch.save(model_l.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best epe loss is: %.5f'%(best_epe))
        logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader):

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, _ = data  
        
        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, pc1, pc2 = model(pos1, pos2, norm1, norm2)

            # eval_loss, chamfer_loss, curvature_loss, smoothness_loss = multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows)
            eval_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

            epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim = 2).mean()

        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_eval = np.mean(metrics['eval_loss'])

    return mean_epe3d, mean_eval
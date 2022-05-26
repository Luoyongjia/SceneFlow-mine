from tools import *

scale = 1.0


def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points_gather(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    # conf GT and loss
    
    # diff_flow = pred_flows[0].permute(0,2,1) - gt_flows[0]
    # diff_flow_l = torch.norm(diff_flow, dim=-1)
    # conf = torch.lt(diff_flow_l,0.07)
    
    # x = torch.tensor(1., device=diff_flow_l.device)
    # y = torch.tensor(0., device=diff_flow_l.device)
    # confmap = torch.where(diff_flow_l<0.07, y, x)
    # BCE_Loss = nn.BCELoss()
    # conf_BCE = BCE_Loss(conf_map, confmap)
    
    # total_loss = 200.0*conf_BCE

    return total_loss
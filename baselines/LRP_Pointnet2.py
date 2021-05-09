import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from attack import CrossEntropyAdvLoss, LogitsAdvLoss

import sys
sys.path.append('../')
sys.path.append('./')

import LRP_utils as utils
from config import BEST_WEIGHTS
from config import MAX_DROP_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from attack import ClipPointsL2, ClipPointsLinf

from model.pointnet2 import farthest_point_sample, square_distance, index_points, query_ball_point


def LRP_scores(model, x, points, label, target):
    # stn layers
    B, Cp, Np = x.shape

    # sa layers
    sa1, sa2, sa3 = model.module.sa1, model.module.sa2, model.module.sa3
    sa1_layers = ['transpose', 'sampling', 'transpose', nn.Sequential(sa1.mlp_convs[0], sa1.mlp_bns[0]), nn.ReLU(),
                  nn.Sequential(sa1.mlp_convs[1], sa1.mlp_bns[1]), nn.ReLU(), nn.Sequential(sa1.mlp_convs[2], sa1.mlp_bns[2]), nn.ReLU(), 'MaxPool']
    sa1_len = len(sa1_layers)
    sa1_x_idx, sa1_point_idx = None, None
    sa1_x = [x] + [None] * sa1_len
    sa1_point = [None] + [None] * sa1_len
    for i in range(sa1_len):
        if sa1_layers[i] == 'transpose':
            sa1_x[i + 1] = sa1_x[i].transpose(1, 2)
            if sa1_point[i] is not None:
                if len(sa1_point[i].shape) > 3:
                    sa1_point[i + 1] = sa1_point[i].transpose(1, 3)
                else:
                    sa1_point[i + 1] = sa1_point[i].transpose(1, 2)
        elif sa1_layers[i] == 'sampling':
            B, _, C = sa1_x[i].shape
            sa1_x_idx = farthest_point_sample(sa1_x[i], sa1.npoint)
            sa1_x[i + 1] = index_points(sa1_x[i], sa1_x_idx)

            sa1_point_idx = query_ball_point(sa1.radius, sa1.nsample, sa1_x[i], sa1_x[i + 1])
            sa1_point[i + 1] = index_points(sa1_x[i], sa1_point_idx) - sa1_x[i + 1].view(B, sa1.npoint, 1, C)
        elif sa1_layers[i] == 'MaxPool':
            sa1_x[i+1] = sa1_x[i]
            sa1_point[i + 1] = torch.max(sa1_point[i], dim=-2)[0]
        else:
            sa1_x[i + 1] = sa1_x[i]
            sa1_point[i + 1] = sa1_layers[i].forward(sa1_point[i])

    # sa2_layers
    sa2_layers = ['transpose', 'sampling', 'cat', 'transpose', nn.Sequential(sa2.mlp_convs[0], sa2.mlp_bns[0]), nn.ReLU(),
                  nn.Sequential(sa2.mlp_convs[1], sa2.mlp_bns[1]), nn.ReLU(), nn.Sequential(sa2.mlp_convs[2], sa2.mlp_bns[2]), nn.ReLU(), 'MaxPool']

    sa2_len = len(sa2_layers)
    sa2_x_idx, sa2_point_idx = None, None
    sa2_x = [sa1_x[-1]] + [None] * sa2_len
    sa2_point = [sa1_point[-1]] + [None] * sa2_len
    sa2_g_x = None
    for i in range(sa2_len):
        if sa2_layers[i] == 'transpose':
            sa2_x[i + 1] = sa2_x[i].transpose(2, 1)
            if sa2_point[i] is not None:
                if len(sa2_point[i].shape) > 3:
                    sa2_point[i + 1] = sa2_point[i].transpose(1, 3)
                else:
                    sa2_point[i + 1] = sa2_point[i].transpose(1, 2)
        elif sa2_layers[i] == 'sampling':
            B, _, C = sa2_x[i].shape
            sa2_x_idx = farthest_point_sample(sa2_x[i], sa2.npoint)
            sa2_x[i + 1] = index_points(sa2_x[i], sa2_x_idx)

            sa2_point_idx = query_ball_point(sa2.radius, sa2.nsample, sa2_x[i], sa2_x[i + 1])
            sa2_g_x = index_points(sa2_x[i], sa2_point_idx) - sa2_x[i + 1].view(B, sa2.npoint, 1, C)
            sa2_point[i + 1] = index_points(sa2_point[i], sa2_point_idx)
        elif sa2_layers[i] == 'cat':
            sa2_x[i + 1] = sa2_x[i]
            sa2_point[i + 1] = torch.cat([sa2_g_x, sa2_point[i]], dim=-1)
        elif sa2_layers[i] == 'MaxPool':
            sa2_x[i + 1] = sa2_x[i]
            sa2_point[i + 1] = torch.max(sa2_point[i], dim=-2)[0]
        else:
            sa2_x[i + 1] = sa2_x[i]
            sa2_point[i + 1] = sa2_layers[i].forward(sa2_point[i])

    #  sa3 layers
    sa3_layers = ['transpose', 'sampling', 'cat', 'transpose', nn.Sequential(sa3.mlp_convs[0], sa3.mlp_bns[0]), nn.ReLU(),
                  nn.Sequential(sa3.mlp_convs[1], sa3.mlp_bns[1]), nn.ReLU(), nn.Sequential(sa3.mlp_convs[2], sa3.mlp_bns[2]), nn.ReLU(), 'MaxPool']

    sa3_len = len(sa3_layers)
    sa3_x_idx, sa3_point_idx = None, None
    sa3_x = [sa2_x[-1]] + [None] * sa3_len
    sa3_point = [sa2_point[-1]] + [None] * sa3_len
    sa3_g_x = None
    for i in range(sa3_len):
        if sa3_layers[i] == 'transpose':
            if sa3_x[i] is not None:
                sa3_x[i + 1] = sa3_x[i].transpose(1, 2)
            if sa3_point[i] is not None:
                if len(sa3_point[i].shape) > 3:
                    sa3_point[i + 1] = sa3_point[i].transpose(1, 3)
                else:
                    sa3_point[i + 1] = sa3_point[i].transpose(1, 2)
        elif sa3_layers[i] == 'sampling':
            B, N, C = sa3_x[i].shape
            sa3_x[i + 1] = sa3_x[i].view(B, 1, N, C)
            sa3_point[i + 1] = sa3_point[i].view(B, 1, N, -1)
        elif sa3_layers[i] == 'cat':
            sa3_point[i + 1] = torch.cat([sa3_x[i], sa3_point[i]], dim=-1)
        elif sa3_layers[i] == 'MaxPool':
            sa3_x[i + 1] = sa3_x[i]
            sa3_point[i + 1] = torch.max(sa3_point[i], dim=-2)[0]
        else:
            sa3_x[i + 1] = sa3_x[i]
            sa3_point[i + 1] = sa3_layers[i].forward(sa3_point[i])

    # classifier layers
    cls_layers = [nn.Sequential(model.module.fc1, model.module.bn1), nn.ReLU(), nn.Sequential(model.module.fc2, model.module.bn2), nn.ReLU(), model.module.fc3]
    cls_layers = utils.toconv(cls_layers)
    cls_len = len(cls_layers)
    cls_x = [sa3_point[-1]] + [None] * cls_len
    for i in range(cls_len):
        cls_x[i + 1] = cls_layers[i].forward(cls_x[i])


    cls_x[-1] = F.softmax(cls_x[-1], dim=1)

    # ***************************************************LRP*****************************************************
    T = torch.arange(0, 40, dtype=torch.long).view(1, -1, 1).cuda()
    T1 = target.view(-1, 1, 1) == T
    T2 = label.view(-1, 1, 1) == T
    bids = torch.arange(0, B)

    cls_fR = -cls_x[-1][bids, label, :].unsqueeze(-2) * cls_x[-1] + cls_x[-1]*T2

    # cls layers
    cls_R = [None] * cls_len + [cls_x[-1] * T1]
    cls_R[-1] = cls_R[-1] / torch.max(cls_R[-1], dim=1, keepdim=True)[0]
    for i in range(0, cls_len)[::-1]:
        if isinstance(cls_layers[i], nn.ReLU):
            cls_R[i] = cls_R[i + 1]
        elif isinstance(cls_layers[i], nn.Sequential):
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(cls_layers[i], rho).forward(cls_x[i]))  # step 1
            bn_weight = cls_layers[i][1].weight / (cls_layers[i][1].running_var.sqrt() + 1e-9)
            while len(bn_weight.shape) < len(cls_layers[i][0].weight.shape):
                bn_weight = bn_weight.unsqueeze(-1)
            W = rho(cls_layers[i][0].weight * bn_weight)

            W = (W.data).unsqueeze(0)
            s = cls_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            cls_R[i] = cls_x[i] * c
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(cls_layers[i], rho).forward(cls_x[i]))  # step 1
            W = rho(cls_layers[i].weight)

            W = (W.data).unsqueeze(0)
            s = cls_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            cls_R[i] = cls_x[i] * c

    # sa layers
    # sa3 layer
    sa3_PR = [None] * sa3_len + [cls_R[0]]
    sa3_XR = [None] * sa3_len + [None]
    for i in range(0, sa3_len)[::-1]:
        if sa3_layers[i] == 'transpose':
            if sa3_x[i] is not None:
                sa3_XR[i] = sa3_XR[i + 1].transpose(1, 2)
            if sa3_point[i] is not None:
                if len(sa3_point[i].shape) > 3:
                    sa3_PR[i] = sa3_PR[i + 1].transpose(1, 3)
                else:
                    sa3_PR[i] = sa3_PR[i + 1].transpose(1, 2)
        elif sa3_layers[i] == 'MaxPool':
            # max_id = torch.max(feat_x[i], dim=-1)[1]
            # feat_R[i] = torch.zeros_like(feat_x[i])
            # batch_ids = torch.arange(0, x.shape[0]).view(-1, 1)
            # fids = torch.arange(0, 1024).view(1, -1)
            # feat_R[i][batch_ids, fids, max_id] = feat_R[i + 1][:, :, 0]
            distri = sa3_x[i] / (torch.sum(sa3_x[i], dim=-2, keepdim=True) + 1e-10)
            sa3_PR[i] = distri * sa3_PR[i+1]
        elif sa3_layers[i] == 'cat':
            sa3_XR[i] = sa3_PR[i + 1][..., 0:3]
            sa3_PR[i] = sa3_PR[i + 1][..., 3:]
        elif sa3_layers[i] == 'sampling':
            B, N, C = sa3_x[i].shape
            sa3_PR[i] = sa3_PR[i + 1].view(B, N, -1)
            sa3_XR[i] = sa3_XR[i + 1].view(B, N, C)
        elif isinstance(sa3_layers[i], nn.ReLU):
            sa3_PR[i] = sa3_PR[i + 1]
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(sa3_layers[i], rho).forward(sa3_x[i]))  # step 1
            bn_weight = sa3_layers[i][1].weight / (sa3_layers[i][1].running_var.sqrt() + 1e-9)
            while len(bn_weight.shape) < len(sa3_layers[i][0].weight.shape):
                bn_weight = bn_weight.unsqueeze(-1)
            W = rho(sa3_layers[i][0].weight * bn_weight)

            W = (W.data).unsqueeze(0)
            s = sa3_PR[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            sa3_PR[i] = sa3_x[i] * c

    # sa2 layer
    sa2_PR = [None] * sa2_len + [sa3_PR[0]]
    sa2_XR = [None] * sa2_len + [sa3_XR[0]]
    for i in range(0, sa2_len)[::-1]:
        if sa2_layers[i] == 'transpose':
            if sa2_x[i] is not None:
                sa2_XR[i] = sa2_XR[i + 1].transpose(1, 2)
            if sa2_point[i] is not None:
                if len(sa2_point[i].shape) > 3:
                    sa2_PR[i] = sa2_PR[i + 1].transpose(1, 3)
                else:
                    sa2_PR[i] = sa2_PR[i + 1].transpose(1, 2)
        elif sa2_layers[i] == 'MaxPool':
            # max_id = torch.max(feat_x[i], dim=-1)[1]
            # feat_R[i] = torch.zeros_like(feat_x[i])
            # batch_ids = torch.arange(0, x.shape[0]).view(-1, 1)
            # fids = torch.arange(0, 1024).view(1, -1)
            # feat_R[i][batch_ids, fids, max_id] = feat_R[i + 1][:, :, 0]
            distri = sa2_x[i] / (torch.sum(sa2_x[i], dim=-2, keepdim=True) + 1e-10)
            sa2_PR[i] = distri * sa2_PR[i + 1]
        elif sa2_layers[i] == 'cat':
            sa2_XR[i] = sa3_PR[i + 1][..., 0:3]
            sa3_PR[i] = sa3_PR[i + 1][..., 3:]
        elif sa3_layers[i] == 'sampling':
            B, N, C = sa3_x[i].shape
            sa3_PR[i] = sa3_PR[i + 1].view(B, N, -1)
            sa3_XR[i] = sa3_XR[i + 1].view(B, N, C)
        elif isinstance(sa3_layers[i], nn.ReLU):
            sa3_PR[i] = sa3_PR[i + 1]
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(sa3_layers[i], rho).forward(sa3_x[i]))  # step 1
            bn_weight = sa3_layers[i][1].weight / (sa3_layers[i][1].running_var.sqrt() + 1e-9)
            while len(bn_weight.shape) < len(sa3_layers[i][0].weight.shape):
                bn_weight = bn_weight.unsqueeze(-1)
            W = rho(sa3_layers[i][0].weight * bn_weight)

            W = (W.data).unsqueeze(0)
            s = sa3_PR[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            sa3_PR[i] = sa3_x[i] * c

    R0 = feat_R[0]
    elif i == 0:
    lb = feat_x[i] * 0 - 1.0
    hb = feat_x[i] * 0 + 1.0
    z = utils.newlayer(feat_layers[i], lambda p: p).forward(feat_x[i]) + 1e-9  # step 1 (a)
    z -= utils.newlayer(feat_layers[i], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
    z -= utils.newlayer(feat_layers[i], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)

    W = feat_layers[i].weight
    rho_plus = lambda p: p.clamp(min=0)
    W_plus = rho_plus(feat_layers[i].weight)
    rho_minus = lambda p: p.clamp(max=0)
    W_minus = rho_minus(feat_layers[i].weight)

    W = (W.data).unsqueeze(0)
    W_plus = (W_plus.data).unsqueeze(0)
    W_minus = (W_minus.data).unsqueeze(0)
    s = feat_R[i + 1] / z
    c = torch.sum(W * s.unsqueeze(-2), dim=1)
    cp = torch.sum(W_plus * s.unsqueeze(-2), dim=1)
    cm = torch.sum(W_minus * s.unsqueeze(-2), dim=1)
    feat_R[i] = c * feat_x[i] - cp * lb - cm * hb

    return R0


def main():
    # build model
    model = PointNet2ClsSsg(num_classes=40)
    model = nn.DataParallel(model).cuda()
    model.eval()

    # load model weight
    print('Loading weight {}'.format(BEST_WEIGHTS['pointnet2']))
    state_dict = torch.load(BEST_WEIGHTS['pointnet2'])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.module.load_state_dict(state_dict)

    # load dataset
    test_set = ModelNet40Attack('data/attack_data.npz', num_points=1024,
                                normalize=True)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=True, num_workers=4,
                             pin_memory=True, drop_last=False)

    ti = 412
    data_iter = iter(test_loader)
    for i in range(ti):
        data = next(data_iter)
    total_num = 0
    success_num = 0
    at_success_num = 0
    i = 0
    all_adv = []
    all_real_label = []
    all_target_label = []
    for x, label, target in tqdm(test_loader):
        # x, label, target = data
        x, label, target = x.cuda(), label.long().cuda(), target.long().cuda()
        x = x.transpose(2, 1).contiguous()
        x.requires_grad = True
        rx = x.clone()

        x_pred = model(x)
        R1 = LRP_scores(model, x, None, label, label)
        x_pred = torch.argmax(x_pred, dim=-1)
        # if x_pred != label:
        #     # all_adv.append(x_adv.transpose(1, 2).contiguous().detach().cpu().numpy())
        #     # all_real_label.append(label.detach().cpu().numpy())
        #     # all_target_label.append(target.detach().cpu().numpy())
        #     continue

        total_num += x.shape[0]
        x_adv = AOA_Attack(model, x, label, target)
        pred, _, _ = model(x_adv)
        pred = torch.argmax(pred, dim=-1)
        success_num += (pred != label).sum().cpu().item()
        logits_at = model_at(x_adv)
        pred_at = torch.argmax(logits_at, dim=-1)
        at_success_num += (pred_at != label).sum().cpu().item()
        i += 1
        if i % 20 == 0:
            print("current attack success rate is", success_num / total_num)
            print("current pointnet++ attack success rate is", at_success_num / total_num)
        all_adv.append(x_adv.transpose(1, 2).contiguous().detach().cpu().numpy())
        all_real_label.append(label.detach().cpu().numpy())
        all_target_label.append(target.detach().cpu().numpy())
        # if i % 20 == 0:
        #     break
        # R0 = LRP_scores(model, x_adv, label, label)
        # R1 = LRP_scores(model, x, label, label)
        # utils.pc_heatmap(x_adv.transpose(2, 1)[0], R0[0].sum(-2).unsqueeze(-1))

    attacked_data = np.concatenate(all_adv, axis=0)  # [num_data, K, 3]
    real_label = np.concatenate(all_real_label, axis=0)  # [num_data]
    target_label = np.concatenate(all_target_label, axis=0)  # [num_data]
    # save results
    save_path = 'attack/results/{}_{}/AOA/{}'. \
        format('mn40', 1024, 'pointnet')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = '{}-budget_{}-iter_{}' \
                '-success_{:.4f}-rank_{}.npz'. \
        format('aoa', 0.5,
               200, success_num/total_num, 0)
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8))
    print("total attack success rate is", success_num/total_num)
    # utils.pc_heatmap(rx.transpose(2, 1)[0], R0[0].sum(-2).unsqueeze(-1))
    # print(x)


if __name__ == '__main__':
    global BATCH_SIZE, BEST_WEIGHTS
    BATCH_SIZE = BATCH_SIZE[1024]
    BEST_WEIGHTS = BEST_WEIGHTS['mn40'][1024]
    cudnn.benchmark = True
    # attack model
    model_at = PointNetCls(k=40, feature_transform=False)
    model_at = nn.DataParallel(model_at).cuda()
    model_at.eval()
    print('Loading weight {}'.format(BEST_WEIGHTS['pointnet']))
    state_dict = torch.load(BEST_WEIGHTS['pointnet'])
    try:
        model_at.load_state_dict(state_dict)
    except RuntimeError:
        model_at.module.load_state_dict(state_dict)
    main()
    print("End!!!")
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

import sys
sys.path.append('../')
sys.path.append('./')

import LRP_utils as utils
from config import BEST_WEIGHTS
from config import MAX_DROP_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import ClipPointsL2


def LRP_scores(model, x, label, target):
    # stn layers
    B, Cp, Np = x.shape
    iden = torch.from_numpy(np.array([
        1, 0, 0, 0, 1, 0, 0, 0, 1
    ]).astype(np.float32)).view(1, 9).repeat(x.shape[0], 1).cuda()

    # feat layers
    feat = model.module.feat
    feat_layers = [feat.conv1, nn.ReLU(), feat.conv2, nn.ReLU(), feat.conv3, nn.ReLU(), nn.MaxPool1d(Np)]
    feat_len = len(feat_layers)
    feat_x = [x] + [None] * feat_len
    for i in range(feat_len):
        if feat_layers[i] == 'SqueezeLast':
            feat_x[i + 1] = feat_x[i].view(-1, 1024)
        else:
            feat_x[i + 1] = feat_layers[i].forward(feat_x[i])

    # classifier layers
    cls_layers = [model.module.fc1, nn.ReLU(), nn.Sequential(model.module.fc2, model.module.bn2), nn.ReLU(),
                  model.module.fc3]
    cls_layers = utils.toconv(cls_layers)
    cls_len = len(cls_layers)
    cls_x = [feat_x[-1]] + [None] * cls_len
    for i in range(cls_len):
        cls_x[i + 1] = cls_layers[i].forward(cls_x[i])

    cls_x[-1] = F.softmax(cls_x[-1], dim=1)
    # ***************************************************LRP*****************************************************
    T = torch.arange(0, 40, dtype=torch.long).view(1, -1, 1).cuda()
    T1 = target.view(-1, 1, 1) == T
    T2 = label.view(-1, 1, 1) == T

    cls_fR = -cls_x[-1][:, label:label + 1, :] * cls_x[-1] + cls_x[-1]*T2

    # cls layers
    cls_R = [None] * cls_len + [cls_x[-1] * T1]
    # cls_R[-1] = cls_R[-1] / torch.max(cls_R[-1], dim=1)[0]
    for i in range(0, cls_len)[::-1]:
        if isinstance(cls_layers[i], nn.ReLU):
            cls_R[i] = cls_R[i + 1]
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(cls_layers[i], rho).forward(cls_x[i]))  # step 1
            if isinstance(cls_layers[i], nn.Conv1d):
                W = rho(cls_layers[i].weight)
            else:
                W = rho(cls_layers[i][0].weight) * ((cls_layers[i][1].weight /
                           torch.sqrt(cls_layers[i][1].running_var + cls_layers[i][1].eps)).view(-1, 1, 1))

            W = (W.data).unsqueeze(0)
            s = cls_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            cls_R[i] = cls_x[i] * c

    # feat layers
    feat_R = [None] * feat_len + [cls_R[0]]
    for i in range(0, feat_len)[::-1]:
        if feat_layers[i] == 'transpose':
            feat_R[i] = feat_R[i + 1].transpose(2, 1)
        elif isinstance(feat_layers[i], nn.MaxPool1d):
            distri = feat_x[i] / (torch.sum(feat_x[i], dim=-1, keepdim=True) + 1e-10)
            feat_R[i] = distri * feat_R[i+1]
        elif isinstance(feat_layers[i], nn.ReLU):
            feat_R[i] = feat_R[i + 1]
        elif i == 0:
            lb = feat_x[i] * 0 - 1.5
            hb = feat_x[i] * 0 + 1.5
            z = feat_layers[i].forward(feat_x[i]) + 1e-9  # step 1 (a)
            z -= utils.newlayer(feat_layers[i], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
            z -= utils.newlayer(feat_layers[i], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)

            W = feat_layers[i][0].weight * ((feat_layers[i][1].weight /
                                            torch.sqrt(feat_layers[i][1].running_var + feat_layers[i][1].eps)).view(-1, 1, 1))
            rho_plus = lambda p: p.clamp(min=0)
            W_plus = rho_plus(feat_layers[i][0].weight) * ((feat_layers[i][1].weight /
                                            torch.sqrt(feat_layers[i][1].running_var + feat_layers[i][1].eps)).view(-1, 1, 1))
            rho_minus = lambda p: p.clamp(max=0)
            W_minus = rho_minus(feat_layers[i][0].weight) * ((feat_layers[i][1].weight /
                                            torch.sqrt(feat_layers[i][1].running_var + feat_layers[i][1].eps)).view(-1, 1, 1))

            W = (W.data).unsqueeze(0)
            W_plus = (W_plus.data).unsqueeze(0)
            W_minus = (W_minus.data).unsqueeze(0)
            s = feat_R[i+1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            cp = torch.sum(W_plus * s.unsqueeze(-2), dim=1)
            cm = torch.sum(W_minus * s.unsqueeze(-2), dim=1)
            feat_R[i] = c * feat_x[i] - cp * lb - cm * hb
        else:
            rho = lambda p: p.clamp(min=0)
            incr = lambda y: y + 1e-9
            z = incr(utils.newlayer(feat_layers[i], rho).forward(feat_x[i]))  # step 1
            if isinstance(feat_layers[i], nn.Conv1d):
                W = rho(feat_layers[i].weight)
            else:
                W = rho(feat_layers[i][0].weight) * ((feat_layers[i][1].weight /
                            torch.sqrt(feat_layers[i][1].running_var + feat_layers[i][1].eps)).view(-1, 1, 1))

            W = (W.data).unsqueeze(0)
            s = feat_R[i + 1] / z
            c = torch.sum(W * s.unsqueeze(-2), dim=1)
            feat_R[i] = feat_x[i] * c

    R0 = feat_R[0]

    return R0


def AOA_Attack(model, x_adv, label, target):
    clip_func = ClipPointsL2(budget=0.5)
    x_ori = x_adv.clone().detach()
    eps = 0.04
    N = x_adv.shape[0] * x_adv.shape[1] * x_adv.shape[2]
    alpha = 0.00016
    iter_k = 0
    lmd = 10
    torch.autograd.set_detect_anomaly(True)
    while iter_k < 200:
        pred, _, _ = model(x_adv)
        lsec = torch.topk(pred, k=2, dim=-1)[1].squeeze()
        # largest = torch.topk(pred, k=2, dim=-1)
        # print(pred[:, label].item(), pred[:, target].item())
        R_ori = LRP_scores(model, x_adv, label, label)
        R_sec = LRP_scores(model, x_adv, label, lsec[1])
        R_ori_max = R_ori.abs().max() + 1e-9
        R_sec_max = R_sec.abs().max() + 1e-9
        if iter_k > 1:
            log_loss = torch.log(R_ori.abs().sum()) - torch.log(R_sec.abs().sum())
            # log_loss = -(R_ori / R_ori_max - R_sec / R_sec_max).abs().sum()
        else:
            log_loss = R_sec.sum() #  - R_sec.sum()
        # log_loss = torch.log(R_ori.abs().sum())   #  - torch.log(R_sec.abs().sum())
        #print(pred[:, label], pred[:, lsec[1]])
        pred = torch.argmax(pred, dim=-1)
        if pred != label:
            break
        # ce_loss = nn.CrossEntropyLoss()(pred, target)
        loss = log_loss      # + lmd * ce_loss
        loss.backward()
        g = x_adv.grad
        perturbation = alpha * g / (g.abs().sum() / N)
        # with torch.no_grad():
        x_adv = x_adv - perturbation
        # x_adv = x_adv.clamp(min=-1, max=1)
        x_adv = x_ori + (x_adv-x_ori).clamp(min=-eps, max=eps)
        # x_adv = x_adv.clamp(min=-1, max=1)
        x_adv = clip_func(x_adv, x_ori)
        x_adv = (x_adv.data).requires_grad_(True)
        iter_k += 1
        # if iter_k%40==0:
        #     alpha *= 0.1
        # print(loss)

    print("iter time is%d" %iter_k)
    return x_adv


def Drop(model, x_adv, label, target):
    N = x_adv.shape[2]
    for i in range(50):
        R_ori = LRP_scores(model, x_adv, label, label).sum(-2).squeeze()
        kids = torch.topk(R_ori, k=N - 1, dim=-1, largest=False)[1]
        # x = torch.zeros_like(x_adv)
        x = x_adv[:, :, kids]
        x_adv = x
        N = x_adv.shape[2]
    return x_adv


def RMSE(xori, xadv):
    x = xadv-xori
    # N = x.shape[0] * x.shape[1] * x.shape[2]
    return torch.sqrt(torch.sum(x**2))

def main():
    global BATCH_SIZE, BEST_WEIGHTS
    BATCH_SIZE = BATCH_SIZE[1024]
    BEST_WEIGHTS = BEST_WEIGHTS['mn40'][1024]
    cudnn.benchmark = True

    # build model
    model = PointNetCls(k=40, feature_transform=False)
    model = nn.DataParallel(model).cuda()
    model.eval()

    # load model weight
    print('Loading weight {}'.format(BEST_WEIGHTS['pointnet']))
    state_dict = torch.load(BEST_WEIGHTS['pointnet'])
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.module.load_state_dict(state_dict)

    # load dataset
    test_set = ModelNet40Attack('data/attack_data.npz', num_points=1024,
                                normalize=True)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False)

    ti = 4
    data_iter = iter(test_loader)
    for i in range(ti):
        data = next(data_iter)
    total_num = 0
    success_num = 0
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

        x_pred, _, _ = model(x)
        x_pred = torch.argmax(x_pred, dim=-1)
        if x_pred != label:
            continue

        total_num += 1
        x_adv = AOA_Attack(model, x, label, target)
        # x_adv = Drop(model, x, label, target)
        pred, _, _ = model(x_adv)
        pred = torch.argmax(pred, dim=-1)
        if pred != label:
            success_num += 1
        i += 1
        if i % 50 == 0:
            print("current attack success rate is", success_num / total_num)
        all_adv.append(x_adv.transpose(1, 2).contiguous().detach().cpu().numpy())
        all_real_label.append(label.detach().cpu().numpy())
        all_target_label.append(target.detach().cpu().numpy())
        # if i % 50 == 0:
        #     break
        # R0 = LRP_scores(model, x_adv, label, label)
        # # R1 = LRP_scores(model, x, label, label)
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
    main()
    print("End!!!")
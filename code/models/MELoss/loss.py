import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def dice_loss_mute(prediction, soft_ground_truth, num_class, weight_map=None):
    pred = prediction.reshape(-1, num_class)
    ground = soft_ground_truth.reshape(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect) / (ref_vol + seg_vol + 1e-5)
    dice_score = 1 - torch.mean(dice_score)

    return dice_score


class DiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=2, weight_map=None, eps=1e-8):
        dice_loss = dice_loss_mute(prediction, soft_ground_truth, num_class, weight_map)

        return dice_loss


class Exclusion_loss(nn.Module):
    def __init__(self, union_func):
        super(Exclusion_loss, self).__init__()
        self.union = union_func

    def forward(self, network_output, target, num_class):
        
        #Intersection between prediction and En is as small as possible!!!(just completely contrary to the dc/crossEntropy loss)
        return -self.union(network_output, target, num_class)


class DC_CE_Marginal_Value_Exclusion_loss(nn.Module):
    def __init__(self, num_classes):
        super(DC_CE_Marginal_Value_Exclusion_loss, self).__init__()
        self.num_classes = num_classes
        
        self.ce = torch.nn.CrossEntropyLoss()
        self.dc = DiceLoss()
        
        self.ex = Exclusion_loss(self.dc)

    def forward(self, net_output, target):
        target_rev = target.long()
        target = F.one_hot(target_rev, self.num_classes).permute(0, 4, 1, 2, 3)
        batch_size = target.shape[0]
        net_output_soft = torch.softmax(net_output, dim=1)
        net_output_soft = torch.clamp(net_output_soft, min=1e-10, max=1)
        result_loss = torch.tensor(0.0, requires_grad=True).cuda()

        for i in range(batch_size):
            target_class = torch.unique(target_rev[i])
            target_class_nozero = target_class[target_class != 0]
            if target_class_nozero.nelement() == 0:
                continue

            target_sm_onehot = torch.zeros([1, len(target_class), *target.shape[2:]]).cuda()
            new_sm_prediction = torch.zeros_like(target_sm_onehot).cuda()
            ex_sm_target = torch.zeros([1, self.num_classes, *target.shape[2:]]).cuda()

            b = 0
            for j in range(self.num_classes):
                if j in target_class:
                    target_sm_onehot[0, b, :, :, :] = target[i, j, :, :, :]
                    new_sm_prediction[0, b, :, :, :] = net_output[i, j, :, :, :]
                    ex_sm_target[0, j, :, :, :] = 1 - target_sm_onehot[0, b, :, :, :]
                    b += 1
                else:
                    new_sm_prediction[0, 0, :, :, :] += net_output[i, j, :, :, :]
                    ex_sm_target[0, j, :, :, :] = 1 - target_sm_onehot[0, 0, :, :, :]
            new_sm_prediction_soft = torch.softmax(new_sm_prediction, dim=1)

            dc_loss = self.dc(new_sm_prediction_soft.permute(0, 2, 3, 4, 1), target_sm_onehot.permute(0, 2, 3, 4, 1), new_sm_prediction.shape[1])
            ce_loss = self.ce(new_sm_prediction, target_sm_onehot)
            ex_loss = self.ex(net_output_soft[i].unsqueeze(0).permute(0, 2, 3, 4, 1), ex_sm_target.permute(0, 2, 3, 4, 1), net_output_soft.shape[1])

            result_loss += 0.5 * ce_loss + 0.5 * dc_loss + 0.1 * ex_loss

        return result_loss/batch_size

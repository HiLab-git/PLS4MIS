import torch
from torch import nn
import numpy as np
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


class DiceLoss_Sample_exp(_Loss):
    def __init__(self, *args, **kwargs):
        super(DiceLoss_Sample_exp, self).__init__()
        
    def forward(self, prediction, soft_ground_truth, num_class=2, weight_map=None, eps=1e-8):
        sample_len = soft_ground_truth.shape[0]
        channel_len = soft_ground_truth.shape[4]
        dice_loss_sum = 0
        
        for i in range(sample_len):
            a = 0
            prediction_sm = prediction[i]
            gt_sm = soft_ground_truth[i]
            for j in range(channel_len):
                if torch.sum(gt_sm[:, :, :, j]) == 0:
                    if j == 0:
                        dice_loss_sp = 0
                    else:
                        dice_loss_sp = dice_loss_mute(prediction_sm[:, :, :, :j], gt_sm[:, :, :, :j], j, weight_map)
                    a += 1
                    break
            if a == 0:
                dice_loss_sp = dice_loss_mute(prediction[i], soft_ground_truth[i], num_class, weight_map)
            dice_loss_sum += dice_loss_sp
        
        dice_loss = dice_loss_sum / sample_len
        
        return dice_loss


class Exclusion_loss(nn.Module):
    def __init__(self, union_func):
        super(Exclusion_loss, self).__init__()
        self.union = union_func

    def forward(self, network_output, target, num_class):
        
        #Intersection between prediction and En is as small as possible!!!(just completely contrary to the dc/crossEntropy loss)
        return -self.union(network_output, target, num_class)
    
def softmax_fun(merged_pre, target_onehot):
    shp_tensor = merged_pre.shape
    merged_pre_list = []
    for i in range(shp_tensor[0]):
        a = 0
        merged_pre_sam = merged_pre[i]
        target_onehot_sam = target_onehot[i]
        for j in range(shp_tensor[1]):
            if torch.sum(target_onehot_sam[j]) == 0:
                merged_pre_sam = torch.softmax(merged_pre_sam[:j, :, :, :], dim=0)
                add_num = shp_tensor[1] - j
                zero_tensor = torch.zeros(add_num, shp_tensor[2], shp_tensor[3], shp_tensor[4]).cuda()
                merged_pre_list.append(torch.cat((merged_pre_sam, zero_tensor), dim=0).unsqueeze(0))
                a += 1
                break
        if a == 0:
            merged_pre_list.append(torch.softmax(merged_pre_sam, dim=0).unsqueeze(0))
    sm_merged_pre = torch.cat(merged_pre_list, dim=0)

    return sm_merged_pre

def partial_onehot(target, default_task):

    shp_tensor = list(target.shape)
    cur_task_list = []
    target_rev = torch.argmax(target, dim=1)
    target_onehot_list = []

    for i in range(shp_tensor[0]):
        a = 0

        target_class = torch.unique(target_rev[i])
        cur_task = []
        for j in target_class:
            if j != 0:
                cur_task.append(default_task[j-1])
        cur_task_list.append(cur_task)

        shp_tensor_new = [1, len(target_class), shp_tensor[2], shp_tensor[3], shp_tensor[4]]
        target_sm_onehot = torch.zeros(shp_tensor_new).cuda()
        for j in target_class:
            target_sm_onehot[0, a, :, :, :] = target[i, j, :, :, :]
            a += 1
        target_onehot_list.append(target_sm_onehot)
    
    large = target_onehot_list[0].shape[1]
    for i in range(1, len(target_onehot_list)):
        if large < target_onehot_list[i].shape[1]:
            large = target_onehot_list[i].shape[1]
    for i in range(len(target_onehot_list)):
        if target_onehot_list[i].shape[1] < large:
            sm_shape = target_onehot_list[i].shape
            add_num = large - target_onehot_list[i].shape[1]
            zero_tensor = torch.zeros(sm_shape[0], add_num, sm_shape[2], sm_shape[3], sm_shape[4]).cuda()
            target_onehot_list[i] = torch.cat((target_onehot_list[i], zero_tensor), dim=1)
    
    target_onehot = torch.cat(target_onehot_list, dim=0)

    return target_onehot, cur_task_list


def merge_prediction(net_output, target_onehot, cur_task_list, default_task):
    '''
        cur_task_list: GT task list
        default_task: net_output task
    '''
    new_prediction = torch.zeros_like(target_onehot)
    if net_output.device.type == "cuda":
        new_prediction = new_prediction.cuda()
    new_prediction[:, 0, :, :, :] = net_output[:, 0, :, :, :]

    for a in range(len(cur_task_list)):
        for i, task in enumerate(default_task):
            if task in cur_task_list[a]:
                j = cur_task_list[a].index(task)
                new_prediction[a, j+1, :, :, :] += net_output[a, i+1, :, :, :]
            else:
                new_prediction[a, 0, :, :, :] += net_output[a, i+1, :, :, :]
  
    return new_prediction


def expand_gt(net_output, target_onehot, cur_task_list, default_task):

    new_gt = torch.zeros_like(net_output)
    if net_output.device.type == "cuda":
        new_gt = new_gt.cuda()
    new_gt[:, 0, :, :, :] = 1 - target_onehot[:, 0, :, :, :]

    for a in range(len(cur_task_list)):
        for i, task in enumerate(default_task):
            if task in cur_task_list[a]:
                j = cur_task_list[a].index(task)
                new_gt[a, i+1, :, :, :] = 1 - target_onehot[a, j+1, :, :, :]
            else:
                new_gt[a, i+1, :, :, :] = 1 - target_onehot[a, 0, :, :, :]

    return new_gt


class DC_CE_Marginal_loss(nn.Module):
    def __init__(self, aggregate="sum", ex=True):
        super(DC_CE_Marginal_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = torch.nn.CrossEntropyLoss()
        self.dc = DiceLoss_Sample_exp()

    def forward(self, net_output, target, default_task):
        
        target_onehot, cur_task_list = partial_onehot(target, default_task)
        merged_pre = merge_prediction(
            net_output, target_onehot, cur_task_list, default_task)
        sm_merged_pre = softmax_fun(merged_pre, target_onehot)
        dc_loss = self.dc(sm_merged_pre.permute(0, 2, 3, 4, 1), target_onehot.permute(0, 2, 3, 4, 1), sm_merged_pre.shape[1])
        ce_loss = self.ce(merged_pre, target_onehot)
        
        if self.aggregate == "sum":
            result = 0.5 * ce_loss + 0.5 * dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            raise NotImplementedError("nah son")

        return result
    

class DC_CE_Marginal_Exclusion_loss(nn.Module):
    def __init__(self, aggregate="sum", ex=True):
        super(DC_CE_Marginal_Exclusion_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = torch.nn.CrossEntropyLoss()
        self.dc = DiceLoss_Sample_exp()
        
        self.ex = Exclusion_loss(self.dc)
        self.ex_choice = ex
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, default_task):
    
        target_onehot, cur_task_list = partial_onehot(target, default_task)
        merged_pre = merge_prediction(
            net_output, target_onehot, cur_task_list, default_task)
        sm_merged_pre = softmax_fun(merged_pre, target_onehot)
        
        not_gt = expand_gt(net_output, target_onehot, cur_task_list, default_task)
        
        dc_loss = self.dc(sm_merged_pre.permute(0, 2, 3, 4, 1), target_onehot.permute(0, 2, 3, 4, 1), sm_merged_pre.shape[1])
        ce_loss = self.ce(merged_pre, target_onehot)

        sm_net_output = torch.softmax(net_output, dim=1)
        ex_loss = self.ex(sm_net_output.permute(0, 2, 3, 4, 1), not_gt.permute(0, 2, 3, 4, 1), net_output.shape[1])
        
        if self.aggregate == "sum":
            result = 0.5 * ce_loss + 0.5 * dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            raise NotImplementedError("nah son")
        if self.ex_choice:
            result = result + 0.1 * ex_loss

        return result    

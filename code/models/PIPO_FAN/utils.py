import torch
from torch import nn
import numpy as np

def partial_onehot_sample(target, cur_task, default_task):

    shp_tensor = list(target.shape)
    target_onehot_list = []

    for i in range(shp_tensor[0]):
        shp_tensor_new = [1, len(cur_task[i])+1, shp_tensor[2], shp_tensor[3], shp_tensor[4]]
        target_sm_onehot = torch.zeros(shp_tensor_new).cuda()
        target_sm_onehot[0, 0, :, :, :] = target[i, 0, :, :, :]
        
        for j, task in enumerate(default_task):
            if task in cur_task[i]:
                a = cur_task[i].index(task)
                target_sm_onehot[0, a+1, :, :, :] = target[i, j+1, :, :, :]
        target_onehot_list.append(target_sm_onehot)
    
    target_onehot = torch.cat(target_onehot_list, dim=0)

    return target_onehot

def merge_prediction(net_output, target_onehot, cur_task_list, default_task):
    '''
        cur_task_list: GT task list
        default_task: net_output task
    '''
    new_prediction = torch.zeros_like(target_onehot)
    if net_output.device.type == "cuda":
        new_prediction = new_prediction.cuda()
    new_prediction[:, 0, :, :, :] = net_output[:, 0, :, :, :]#先把bkg赋值(bkg不属于任何task)

    for a in range(len(cur_task_list)):
        for i, task in enumerate(default_task):
            if task in cur_task_list[a]:
                j = cur_task_list[a].index(task)
                new_prediction[a, j+1, :, :, :] += net_output[a, i+1, :, :, :]
            else:
                new_prediction[a, 0, :, :, :] += net_output[a, i+1, :, :, :]
  
    return new_prediction

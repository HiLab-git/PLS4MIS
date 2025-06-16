import torch
from torch import nn
import numpy as np
from torch.nn.modules.loss import _Loss
from utils.losses import *
from collections import deque

class CrossEntropyLoss3d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss3d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)     


class DFLoss(nn.Module):
    def __init__(self, weight=None):
        super(DFLoss, self).__init__()
        self.weight = weight
        self.criterion = CrossEntropyLoss3d(self.weight).cuda()
        self.dice_criterion = DiceLoss()
    
    def forward(self, net_output, target):

        ce_loss = torch.tensor(0.0, requires_grad = True).cuda()
        dice_loss = torch.tensor(0.0, requires_grad = True).cuda()
        target_rev = torch.argmax(target, dim=1)
        batch_size = target.shape[0]

        for a in range(batch_size):
            cur_task = torch.unique(target_rev[a])
            cur_task = cur_task[cur_task != 0] 
            if cur_task.nelement() == 0:
                continue
            for i in cur_task:
                target_class = target[a, i].unsqueeze(0).long()
                single_class_pre = net_output[a, [0, i]]
                single_class_pre[0] = torch.clamp(1.0 - single_class_pre[1], min=1e-10, max=1.0)

                ce = self.criterion(single_class_pre.unsqueeze(0), target_class)
                label = F.one_hot(target_class, 2)
                dice = self.dice_criterion(single_class_pre.unsqueeze(0).permute(0, 2, 3, 4, 1), label, num_class = 2)
                ce_loss += ce
                dice_loss += dice

        loss = ce_loss + dice_loss
        return loss


class HADFLoss(nn.Module):
    def __init__(self, num_classes, weight=None):
        super(HADFLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.criterion = CrossEntropyLoss3d(self.weight).cuda()
        self.dice_criterion = DiceLoss()
        self.N = 50  # Set the sliding window size N, which can be adjusted as needed
        self.batch_loss_history = deque(maxlen=self.N)

    def forward(self, net_output, target):
        ce_loss = torch.tensor(0.0, requires_grad=True).cuda()
        dice_loss = torch.tensor(0.0, requires_grad=True).cuda()
        target_rev = torch.argmax(target, dim=1)
        batch_size = target.shape[0]
        
        # Used to store the total loss and number of samples for each category in the current batch
        batch_class_total_loss = torch.zeros(self.num_classes).cuda()
        batch_class_counts = torch.zeros(self.num_classes).cuda()

        # Stores the loss and category index of the current batch
        loss_info = []

        for a in range(batch_size):
            cur_task = torch.unique(target_rev[a])
            cur_task = cur_task[cur_task != 0]
            if cur_task.nelement() == 0:
                continue
            for i in cur_task:
                target_class = target[a, i].unsqueeze(0).long()
                single_class_pre = net_output[a, [0, i]]
                single_class_pre[0] = torch.clamp(1.0 - single_class_pre[1], min=1e-10, max=1.0)

                ce = self.criterion(single_class_pre.unsqueeze(0), target_class)
                label = F.one_hot(target_class, 2)
                dice = self.dice_criterion(single_class_pre.unsqueeze(0).permute(0, 2, 3, 4, 1), label, num_class = 2)

                # Accumulate the total loss and number of samples for each category in the current batch
                batch_class_total_loss[i] += dice.detach()
                batch_class_counts[i] += 1

                # Store loss information for subsequent calculations
                loss_info.append((ce, dice, i))

        # Calculate the average loss of each category in the current batch and store it in the batch loss history queue
        batch_class_avg_loss = batch_class_total_loss / (batch_class_counts + 1e-10)  # Avoid division by zero
        self.batch_loss_history.append(batch_class_avg_loss)
        
        # Calculate the average loss for each class over the last N batches
        class_avg_loss = torch.zeros(self.num_classes).cuda()
        class_counts = torch.zeros(self.num_classes).cuda()

        for batch_loss in self.batch_loss_history:
            class_avg_loss += batch_loss
            class_counts += (batch_loss > 0).float()
        
        # Calculate the average loss for each class
        for i in range(self.num_classes):
            if class_counts[i] > 0:
                class_avg_loss[i] /= class_counts[i]
            else:
                class_avg_loss[i] = 0.0
        
        # Get the class index with non-zero mean loss, excluding the background class (assuming index is 0)
        valid_indices = (class_avg_loss[1:] > 0).nonzero(as_tuple=True)[0] + 1
        if valid_indices.nelement() > 0:
            mean_loss = class_avg_loss[valid_indices].mean()
        else:
            mean_loss = 1.0

        class_weights = class_avg_loss / mean_loss
        class_weights = torch.clamp(class_weights, min=0.5)

        for ce, dice, i in loss_info:
            class_weight = class_weights[i]
            ce_loss += ce * class_weight
            dice_loss += dice * class_weight

        loss = ce_loss + dice_loss
        return loss

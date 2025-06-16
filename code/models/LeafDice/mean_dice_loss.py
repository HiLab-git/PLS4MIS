import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-5


class MeanDiceLoss(nn.Module):
    def __init__(self, reduction='mean', squared=True):
        """
        :param reduction: str. Mode to merge the batch of sample-wise loss values.
        :param squared: bool. If True, squared the terms in the denominator of the Dice loss.
        """
        super(MeanDiceLoss, self).__init__()
        self.squared = squared
        self.reduction = reduction

    def _prepare_data(self, input_batch, target):
        num_out_classes = input_batch.size(1)
        # Prepare the batch prediction
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))  # b,s
        pred_proba = F.softmax(flat_input, dim=1)  # b,c,s
        target_proba = F.one_hot(flat_target, num_classes=num_out_classes)
        target_proba = target_proba.permute(0, 2, 1).float()  # b,c,s
        return pred_proba, target_proba


    def forward(self, input_batch, target):
        """
        compute the mean dice loss between input_batch and target.
        :param input_batch: tensor array. This is the output of the deep neural network (before softmax).
        Dimensions should be in the order:
        1D: num batch, num classes, num voxels
        2D: num batch, num classes, dim x, dim y
        3D: num batch, num classes, dim x, dim y, dim z
        :param target: tensor array. This is the ground-truth segmentation.
        Dimensions should be in the order:
        1D: num batch, num voxels or num batch, 1, num voxels
        2D: num batch, dim x, dim y or num batch, 1, dim x, dim y
        3D: num batch, dim x, dim y, dim z or num batch, 1, dim x, dim y, dim z
        :return: tensor array.
        """
        pred_proba, target_proba = self._prepare_data(input_batch, target)

        # Compute the dice for all classes
        num = pred_proba * target_proba  # b,c,s --p*g
        num = torch.sum(num, dim=2)  # b,c
        if self.squared:
            den1 = torch.sum(pred_proba * pred_proba, dim=2)  # b,c
            den2 = torch.sum(target_proba * target_proba, dim=2)  # b,c
        else:
            den1 = torch.sum(pred_proba, dim=2)  # b,c
            den2 = torch.sum(target_proba, dim=2)  # b,c

        # I choose not to add an epsilon on the numerator.
        # This is because it can lead to unstabilities in the gradient.
        # As a result, for empty prediction and empty target the Dice loss value
        # is 1 and not 0. But it is fine because for the optimization of a
        # deep neural network it is the gradient not the loss that is relevant.
        dice = (2. * num) / (den1 + den2 + EPSILON)

        # Get the mean of the dices over all classes
        loss = 1. - torch.mean(dice, dim=1)  # b,

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        # Default is mean reduction
        else:
            return loss.mean()


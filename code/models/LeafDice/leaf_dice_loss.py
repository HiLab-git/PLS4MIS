import torch
import torch.nn.functional as F
from models.LeafDice.mean_dice_loss import MeanDiceLoss


class LeafDiceLoss(MeanDiceLoss):
    def __init__(self, labels_superset_class, reduction='mean', squared=True):
        """
        :param labels_superset_map: superclasses
        :param reduction: str.
        :param squared: bool.
        """
        super(LeafDiceLoss, self).__init__(
            reduction=reduction,
            squared=squared,
        )
        self.labels_superset_class = labels_superset_class

    def _prepare_data(self, input_batch, target):
        num_out_classes = input_batch.size(1)
        num_total_classes = self.labels_superset_class + 1  # total number of classes (with supersets)
        
        # Prepare the batch prediction
        flat_input = torch.reshape(input_batch, (input_batch.size(0), input_batch.size(1), -1))  # b,c,s
        flat_target = torch.reshape(target, (target.size(0), -1))
        pred_proba = F.softmax(flat_input, dim=1)  # b,c,s
        flat_target = flat_target.long()  # make sure that the target is a long before using one_hot
        target_proba = F.one_hot(
            flat_target, num_classes=num_total_classes).permute(0, 2, 1).float()
       
        # Remove the supersets channels from the target proba.
        # As a consequence they will be masked in the loss
        # no need to do anything else.
        target_proba = target_proba[:, :num_out_classes, :]
        
        return pred_proba, target_proba
    

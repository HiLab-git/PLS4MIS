import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleClassAdaptiveLoss(nn.Module):
    """
    Sample-wise Dataset & Class Adaptive Loss

    - Uses per-class Sigmoid + BCE + Dice loss.
    - Calculates loss only on classes that appear in the given sample (closer to clinical annotation scenarios).
    - Similar to MultiTalent's Eq.(1)(2), but 1_c^(k) is determined by the sample itself:
    m_{b,c} = 1  <=>  at least one voxel in target[b, c] is 1
    """
    def __init__(self,
                 average_over_classes: bool = False,
                 ignore_background: bool = True,
                 eps: float = 1e-5):
        """
        average_over_classes:
            True  -> Divide by the number of valid classes (the number of classes actually involved in the calculation in the batch)
            False -> Sum by class (more like the original text)

        ignore_background:
            True  -> Do not apply BCE+Dice to class 0 (background).
            False -> Treat the background as a separate "class" and apply a Sigmoid (binary) function.

        eps:
            Dice Smoothing term.
        """
        super().__init__()
        self.average_over_classes = average_over_classes
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor):
        """
        logits : (B, C, D, H, W)  —— raw logits (no Sigmoid)
        target : (B, D, H, W) — Integer labels (0..C-1)
                  or (B, C, D, H, W) — Multi-channel 0/1 / soft labels
        """
        device = logits.device
        B, C, *spatial = logits.shape

        # ---- 1. Processing the target: Unifying to the format (B, C, D, H, W) ----
        if target.dim() == 4:
            target_idx = target.long().to(device)           # (B, D, H, W)
            target_oh = F.one_hot(target_idx, num_classes=C).permute(
                0, 4, 1, 2, 3
            ).float()                                                  # (B, C, D, H, W)
        else:
            target_oh = target.float().to(device)                      # (B, C, D, H, W)

        # ---- 2. per-class Sigmoid ----
        prob = torch.sigmoid(logits)                                   # (B, C, D, H, W)

        # ---- 3. Flattening the spatial dimensions ----
        N = 1
        for s in spatial:
            N *= s

        logits_flat = logits.view(B, C, N)         # (B, C, N) for BCE-with-logits
        prob_flat = prob.view(B, C, N)         # (B, C, N)
        target_flat = target_oh.view(B, C, N)  # (B, C, N)

        # ---- 4. sample-wise class mask m_{b,c} ----
        # For each sample/class: If any voxel equals 1, then that class is considered "labeled" in that sample.
        # -> (B, C)
        sample_class_mask = (target_flat > 0.5).any(dim=-1).float()

        if self.ignore_background and C > 0:
            # Class 0 is considered the background and is not included in the Sigmoid-BCE-Dice calculation.
            sample_class_mask[:, 0] = 0.0

        # Extending to the voxel dimension: (B, C, N)
        mask = sample_class_mask.unsqueeze(-1).expand(-1, -1, N)

        # Which classes are labeled by at least one sample in the entire batch? (C,)
        valid_per_class = (mask.sum(dim=(0, 2)) > 0).float()

        # If the entire batch contains no foreground annotations (an extreme case), return 0 directly.
        if valid_per_class.sum() == 0:
            return (logits * 0).sum()

        # ---- 5. BCE Section (by category) ----
        bce_per_voxel = F.binary_cross_entropy_with_logits(
            logits_flat, target_flat, reduction='none'   # (B, C, N)
        )
        bce_weighted = bce_per_voxel * mask           # Only effective for samples with the specified sample-class.

        denom_bce = mask.sum(dim=(0, 2)).clamp_min(1.0)  # Number of valid voxels in each category
        bce_per_class = bce_weighted.sum(dim=(0, 2)) / denom_bce  # (C,)

        # ---- 6. Dice Score Section (Statistics by class, batch level) ----
        intersection = (prob_flat * target_flat * mask).sum(dim=(0, 2))  # (C,)
        pred_sum = (prob_flat * mask).sum(dim=(0, 2))                    # (C,)
        target_sum = (target_flat * mask).sum(dim=(0, 2))                # (C,)

        dice_score = (2.0 * intersection + self.eps) / \
                     (pred_sum + target_sum + self.eps)
        dice_loss_per_class = 1.0 - dice_score                           # (C,)

        # ---- 7. Class Dimension Aggregation ----
        per_class_loss = (bce_per_class + dice_loss_per_class) * valid_per_class  # (C,)
        total_loss = per_class_loss.sum()

        if self.average_over_classes:
            num_valid = valid_per_class.sum().clamp_min(1.0)
            total_loss = total_loss / num_valid

        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss3d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss3d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)


class TAL_CE_Loss(nn.Module):
    """
    Target Adaptive Loss (TAL) style cross-entropy implementation:
    - For the current sample:
    Channel 0: non-target (original background + sum of logits of all unlabeled organs)
    Channels 1..K: actual foreground organs appearing in this sample
    - Labels are also rearranged in this K+1 dimensional space and supervised by CrossEntropyLoss.
    """
    def __init__(self, num_classes: int = 17):
        super().__init__()
        self.num_classes = num_classes
        self.ce = CrossEntropyLoss3d().cuda()

    def forward(self, net_output: torch.Tensor, target_onehot: torch.Tensor, cur_task_batch):
        """
        net_output: (B, C_full, D, H, W)
        target_onehot: (B, C_full, D, H, W)
        """
        net_output = torch.softmax(net_output, dim=1)
        net_output = torch.clamp(net_output, min=1e-10, max=1)
        device = net_output.device
        B, C_full, *spatial = net_output.shape
        assert C_full == self.num_classes, \
            f"C_full({C_full}) != num_classes({self.num_classes})"

        total_loss = (net_output * 0).sum()
        valid = 0

        for i in range(B):
            cur_task = cur_task_batch[i]  # shape: [num_classes-1]
            fg = torch.nonzero(cur_task, as_tuple=False).squeeze(1)
            if fg.numel() == 0:
                continue
            fg += 1

            K = fg.numel()

            # new_logits/new_target: (1, 1+K, D, H, W)
            new_logits = net_output.new_zeros(1, K + 1, *spatial)
            new_target = target_onehot.new_zeros(1, K + 1, *spatial)

            # Channel 0: non-target, copy the original background first.
            new_logits[0, 0] = net_output[i, 0]
            new_target[0, 0] = target_onehot[i, 0]

            # Channel 1..K: Organs that actually appeared in this sample
            for j, c in enumerate(fg):
                new_logits[0, j + 1] += net_output[i, c]
                new_target[0, j + 1] = target_onehot[i, c]

            # All remaining unseen organ pathways are merged into the non-target logit.
            all_fg = torch.arange(1, C_full, device=device)
            absent = all_fg[~torch.isin(all_fg, fg)]
            if absent.numel() > 0:
                new_logits[0, 0] += net_output[i, absent].sum(dim=0)

            # Integer labels for CE (1, D, H, W)
            ce_target = new_target.argmax(dim=1)

            loss_i = self.ce(new_logits, ce_target)
            total_loss = total_loss + loss_i
            valid += 1

        if valid == 0:
            return total_loss

        return total_loss / valid
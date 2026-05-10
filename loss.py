import torch
import torch.nn.functional as F

class LossFunction():
    def __init__(self, smooth=1.0, weight=None, pos_weight=None, reduction="mean"):
        self.smooth = smooth
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
    def __call__(self, pred, label):
        label = label.float()
        if label.dim() == 3:
            label = label.unsqueeze(1)
        loss_bce = F.binary_cross_entropy_with_logits(pred, label, weight=self.weight, pos_weight=self.pos_weight, reduction=self.reduction)
       
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        label = label.flatten(1)

        intersection = (pred * label).sum(dim=1)
        denom = pred.sum(dim=1) + label.sum(dim=1)
        loss_dice = 1.0 - ((2.0 * intersection + self.smooth)/(denom + self.smooth))
        if self.reduction == "mean":
            loss_dice = loss_dice.mean()
        elif self.reduction == "sum":
            loss_dice = loss_dice.sum()
        total_loss = loss_bce + loss_dice
        return total_loss
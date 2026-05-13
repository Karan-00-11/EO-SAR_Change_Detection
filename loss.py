import torch
import torch.nn.functional as F


def _prepare_label(label):
    label = label.float()
    if label.dim() == 3:
        label = label.unsqueeze(1)
    return label


def _reduce_loss(loss, reduction):
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _focal_loss_with_logits(
    logits,
    label,
    alpha,
    gamma,
    weight,
    pos_weight,
):
    prob = torch.sigmoid(logits)
    p_t = prob * label + (1.0 - prob) * (1.0 - label)

    log_prob_pos = F.logsigmoid(logits)
    log_prob_neg = F.logsigmoid(-logits)
    if pos_weight is not None:
        log_prob_pos = log_prob_pos * pos_weight

    log_prob = label * log_prob_pos + (1.0 - label) * log_prob_neg
    loss = -log_prob

    modulating = (1.0 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = alpha * label + (1.0 - alpha) * (1.0 - label)
        modulating = modulating * alpha_t

    loss = loss * modulating
    if weight is not None:
        loss = loss * weight
    return loss

class LossFunction():
    def __init__(self, smooth=1.0, weight=1.0, pos_weight=4.0, reduction="none"):
        self.smooth = smooth
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
    def __call__(self, pred, label):
        label = label.float()
        if label.dim() == 3:
            label = label.unsqueeze(1)
        weight = _as_loss_weight(self.weight, pred)
        pos_weight = _as_loss_weight(self.pos_weight, pred)
        loss_bce = F.binary_cross_entropy_with_logits(
            pred,
            label,
            weight=weight,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )
        loss_bce = loss_bce.mean(dim=(1, 2, 3))  # -> (B,)

       
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
        total_loss = 0.5 * loss_bce + 0.5 * loss_dice
        return total_loss.mean()


class FocalLoss():
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        weight=None,
        pos_weight=4.0,
        reduction="mean",
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def __call__(self, logits, label):
        label = _prepare_label(label)
        loss = _focal_loss_with_logits(
            logits,
            label,
            alpha=self.alpha,
            gamma=self.gamma,
            weight=self.weight,
            pos_weight=self.pos_weight,
        )
        return _reduce_loss(loss, self.reduction)


class BCEFocalLoss():
    def __init__(
        self,
        bce_weight=0.5,
        focal_weight=0.5,
        alpha=0.25,
        gamma=2.0,
        weight=1.0,
        pos_weight=4.0,
        reduction="mean",
    ):
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def __call__(self, logits, label):
        label = _prepare_label(label)
        weight = _as_loss_weight(self.weight, logits)
        pos_weight = _as_loss_weight(self.pos_weight, logits)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            label,
            weight=weight,
            pos_weight=pos_weight,
            reduction="none",
        )
        focal = _focal_loss_with_logits(
            logits,
            label,
            alpha=self.alpha,
            gamma=self.gamma,
            weight=weight,
            pos_weight=pos_weight,
        )
        loss = self.bce_weight * bce + self.focal_weight * focal
        return _reduce_loss(loss, self.reduction)


def _as_loss_weight(value, tensor):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device)


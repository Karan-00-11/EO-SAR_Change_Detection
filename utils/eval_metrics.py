import torch

def iou_score(logits, target):
    target_mask = target.unsqueeze(1) if target.dim() == 3 else target
    pred_mask = (torch.sigmoid(logits) > 0.5)
    intersection = (pred_mask & target_mask.bool()).float().sum((1, 2, 3))
    union = (pred_mask | target_mask.bool()).float().sum((1, 2, 3)).clamp_min(1)
    iou = (intersection / union).mean().item()
    return iou


def precision_recall(logits, target):
    logits = (torch.sigmoid(logits) > 0.5)
    target = target.unsqueeze(1) if target.dim() == 3 else target
    target = target.bool()
    tp = (logits & target).sum((1, 2, 3))
    fp = (logits & ~target).sum((1, 2, 3))
    fn = (~logits & target).sum((1, 2, 3))

    precision = (tp/(tp + fp + 1e-3)).mean().item()
    recall = (tp/(tp + fn + 1e-3)).mean().item()
    return precision, recall

def f1_score(precision, recall):
    return (2*precision*recall)/(precision + recall + 1e-3)
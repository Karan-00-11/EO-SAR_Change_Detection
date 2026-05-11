import torch


def _prep_masks(logits, target, threshold):
    target_mask = target.unsqueeze(1) if target.dim() == 3 else target
    pred_mask = (torch.sigmoid(logits) > threshold)
    return pred_mask, target_mask.bool()


def iou_score(logits, target, threshold=0.5):
    pred_mask, target_mask = _prep_masks(logits, target, threshold)
    intersection = (pred_mask & target_mask).float().sum((1, 2, 3))
    union = (pred_mask | target_mask).float().sum((1, 2, 3)).clamp_min(1)
    iou = (intersection / union).mean().item()
    return iou


def precision_recall(logits, target, threshold=0.5):
    pred_mask, target_mask = _prep_masks(logits, target, threshold)
    tp = (pred_mask & target_mask).sum((1, 2, 3))
    fp = (pred_mask & ~target_mask).sum((1, 2, 3))
    fn = (~pred_mask & target_mask).sum((1, 2, 3))

    precision = (tp / (tp + fp + 1e-3)).mean().item()
    recall = (tp / (tp + fn + 1e-3)).mean().item()
    return precision, recall


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall + 1e-3)


def find_best_threshold(logits, target, metric="f1", thresholds=None):
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, steps=19, device=logits.device)

    target_mask = target.unsqueeze(1) if target.dim() == 3 else target
    target_mask = target_mask.bool().flatten(1)
    probs = torch.sigmoid(logits).flatten(1)

    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred_mask = (probs > t)
        tp = (pred_mask & target_mask).sum(dim=1).float()
        fp = (pred_mask & ~target_mask).sum(dim=1).float()
        fn = (~pred_mask & target_mask).sum(dim=1).float()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        if metric == "iou":
            score = (tp / (tp + fp + fn + 1e-6)).mean().item()
        else:
            score = (2 * precision * recall / (precision + recall + 1e-6)).mean().item()

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score
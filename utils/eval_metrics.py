from pathlib import Path

import torch
import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with _CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def _default_threshold():
    return cfg.get("training", {}).get("threshold", 0.40)


def iou_score(logits, target, threshold=None):
    threshold = _default_threshold() if threshold is None else threshold
    target_mask = target.unsqueeze(1) if target.dim() == 3 else target
    pred_mask = torch.sigmoid(logits) > threshold
    intersection = (pred_mask & target_mask.bool()).float().sum((1, 2, 3))
    union = (pred_mask | target_mask.bool()).float().sum((1, 2, 3)).clamp_min(1)
    return (intersection / union).mean().item()


def precision_recall(logits, target, threshold=None):
    threshold = _default_threshold() if threshold is None else threshold
    pred_mask = torch.sigmoid(logits) > threshold
    target_mask = target.unsqueeze(1) if target.dim() == 3 else target
    target_mask = target_mask.bool()
    tp = (pred_mask & target_mask).sum((1, 2, 3))
    fp = (pred_mask & ~target_mask).sum((1, 2, 3))
    fn = (~pred_mask & target_mask).sum((1, 2, 3))

    precision = (tp / (tp + fp + 1e-3)).mean().item()
    recall = (tp / (tp + fn + 1e-3)).mean().item()
    return precision, recall


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall + 1e-3)
import argparse
import logging
from multiprocessing import freeze_support
import torch
import yaml
import os

from model import CMCDNet
from dataset import ChangeDetctionDataset
from transform import ValTransform
from utils.mask_utils import remap_labels
from torch.utils.data import DataLoader
from loss import LossFunction


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CMCDNet.")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to a checkpoint .pth file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    hyperparameter = cfg["training"]
    validation = cfg["data"]["val"]
    test = cfg["data"]["test"]
    hyperparameter["batch_size"] = 4
    if args.data_path:
        split_root = os.path.join(args.data_path, args.split)
        split_cfg = {
            "pre_event": os.path.join(split_root, "pre-event"),
            "post_event": os.path.join(split_root, "post-event"),
            "target": os.path.join(split_root, "target"),
        }
    else:
        split_cfg = validation if args.split == "val" else test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timm_logger = logging.getLogger("timm")
    previous_timm_level = timm_logger.level
    timm_logger.setLevel(logging.ERROR)
    model = CMCDNet()
    timm_logger.setLevel(previous_timm_level)

    ckpt_path = args.weights
    # print("os.getcwd():", os.getcwd())
    # print("ckpt_path:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Strip common wrapper prefixes from saved checkpoints
    for prefix in ("_orig_mod.", "module."):
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}
            break

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")
        print("Tip: set ckpt_path to a checkpoint saved from the same model definition.")

    criterion = LossFunction()

    eval_dataset = ChangeDetctionDataset(
        pre_img_path=split_cfg["pre_event"],
        post_img_path=split_cfg["post_event"],
        target_img_path=split_cfg["target"],
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["stride"],
        index_path=None,
        build_metadata=False,
        transform=ValTransform(),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=hyperparameter["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # threshold = hyperparameter.get("threshold", 0.20)
    threshold = 0.20

    model = model.to(device)
    model.eval()

    eval_loss_sum = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for pre, post, target, _ in eval_loader:
            pre = pre.to(device)
            post = post.to(device)
            if target.max().item() > 1:
                target = torch.from_numpy(remap_labels(target.cpu().numpy())).to(dtype=target.dtype)
            target = target.to(device)

            logits = model(pre, post)
            loss = criterion(logits, target)
            eval_loss_sum += loss.item()

            pred = (torch.sigmoid(logits) > threshold)
            target_mask = target.unsqueeze(1) if target.dim() == 3 else target
            target_mask = target_mask.bool()
            tp = (pred & target_mask).sum().item()
            fp = (pred & ~target_mask).sum().item()
            fn = (~pred & target_mask).sum().item()
            total_tp += tp
            total_fp += fp
            total_fn += fn

    eval_loss = eval_loss_sum / max(1, len(eval_loader))
    precision = total_tp / (total_tp + total_fp + 1e-3)
    recall = total_tp / (total_tp + total_fn + 1e-3)
    f1 = (2 * precision * recall) / (precision + recall + 1e-3)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-3)

    print(f"{args.split}_loss: {eval_loss:.4f}")
    print(f"precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f} | iou: {iou:.4f}")


if __name__ == "__main__":
    freeze_support()
    main()


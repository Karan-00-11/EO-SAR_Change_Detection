import os

import torch
import wandb as wb
import yaml
import itertools
import time

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from model import CMCDNet
from loss import LossFunction, BCEFocalLoss
from dataset import ChangeDetctionDataset
from transform import PairedTransform, ValTransform
from sampler import BucketBatchSampler
from utils.logging import log_images
from utils.eval_metrics import iou_score, precision_recall, f1_score

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data = cfg["data"]
    hyperparameter = cfg["training"]
    aug = cfg["augmentations"]
    ### Train Dataset
    dataset = ChangeDetctionDataset(
        pre_img_path=data["train"]["pre_event"],
        post_img_path=data["train"]["post_event"],
        target_img_path=data["train"]["target"],
        patch_size=data["patch_size"],
        stride=data["stride"],
        index_path=data["index_path"],
        build_metadata=True,
        transform=PairedTransform(
            horizontal_flip_p=aug["horizontal_flip"]["probability"],
            vertical_flip_p=aug["vertical_flip"]["probability"]
        )
    )

    batch_sampler = BucketBatchSampler(
        dataset.patch_metadata,
        batch_size = hyperparameter["batch_size"],
        ratio=cfg["sampler"]["ratios"],
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,)    ### Validaation Dataset
    ### Validaation Dataset
    validation_dataset = ChangeDetctionDataset(
        pre_img_path=data["val"]["pre_event"],
        post_img_path=data["val"]["post_event"],
        target_img_path=data["val"]["target"],
        patch_size=data["patch_size"],
        stride=data["stride"],
        index_path=None,
        build_metadata=False,
        transform=ValTransform()
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=hyperparameter["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMCDNet().to(device)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")   
        model = torch.compile(model, mode="default") 
        scaler = GradScaler("cuda")
    # criterion = BCEFocalLoss()
    criterion = LossFunction()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameter["learning_rate"],
        weight_decay=hyperparameter["weight_decay"], 
    )

    size = sum(1 for m in dataset.patch_metadata if m["bucket"] != "discard")

    steps_per_epoch = size // hyperparameter["batch_size"] ##15827.875
    print(steps_per_epoch)
    global_step = 0
    # loss_tracker = {}

    for epoch in range(hyperparameter["epochs"]):
        running = 0.0
        pbar = tqdm(itertools.islice(train_loader, steps_per_epoch), total=steps_per_epoch, leave=False)
        for step_idx, (pre, post, target, _) in enumerate(pbar):
            pre = pre.to(device)
            post = post.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(pre, post)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1

            current_step = step_idx + 1
            current_lr = optimizer.param_groups[0]['lr']
            

            pbar.set_description(f"Epoch {epoch+1}")
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            iou_loss = 0.0
            total_tp, total_fp, total_fn = 0, 0, 0
            for pre, post, target, _ in val_loader:
                pre = pre.to(device)
                post = post.to(device)
                target = target.to(device)
                logits = model(pre, post)
                loss = criterion(logits, target)
                val_loss += loss.item()
                iou_loss += iou_score(logits, target)

                pred = (torch.sigmoid(logits) > cfg['training']["threshold"])
                target_mask = target.unsqueeze(1) if target.dim() == 3 else target
                target_mask = target_mask.bool()
                tp = (pred & target_mask).sum().item()
                fp = (pred & ~target_mask).sum().item()
                fn = (~pred & target_mask).sum().item()
                total_tp += tp
                total_fp += fp
                total_fn += fn

            precision = total_tp / (total_tp + total_fp + 1e-3)
            recall = total_tp / (total_tp + total_fn + 1e-3)
            f1 = (2 * precision * recall) / (precision + recall + 1e-3)
            val_avg_loss = val_loss / len(val_loader)
            val_avg_iou = iou_loss / len(val_loader)
            train_epoch_loss = running / steps_per_epoch

            val_log = {
                "val/loss": val_avg_loss,
                "val/iou": val_avg_iou,
                "val/precision": precision,
                "val/recall": recall,
                "val/f1_score": f1,
                "val/learning_rate": current_lr
            }

            ckpt_path = os.path.join(hyperparameter["chk_path"], f"best_epoch{epoch+1}_iou{val_avg_iou:.4f}.pth")
            torch.save({
                "epoch": hyperparameter["epochs"],
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if 'scaler' in globals() else None,
                "best_val_iou": val_avg_iou,
            }, ckpt_path)
            print(
                f"epoch {epoch + 1}/{hyperparameter['epochs']} | "
                f"train_loss {train_epoch_loss:.4f} | "
                f"val_loss {val_avg_loss:.4f} | "
                f"val_iou {val_avg_iou:.4f} | "
                f"val_f1 {f1:.4f} | "
            )
            print()
            print(f"Saved new best checkpoint: {ckpt_path}")

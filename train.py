import torch
import wandb as wb
import yaml
import itertools
import time

from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from model import CMCDNet
from loss import LossFunction
from dataset import ChangeDetctionDataset
from transform import PairedTransform, ValTransform
from sampler import BucketBatchSampler
from utils.logging import log_images
from utils.eval_metrics import iou_score, precision_recall, f1_score

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True



if __name__ == '__main__':
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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)

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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler("cuda")

    model = CMCDNet().to(device)
    criterion = LossFunction()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameter["learning_rate"],
        weight_decay=hyperparameter["weight_decay"], 
    )

    size = sum(1 for m in dataset.patch_metadata if m["bucket"] != "discard")

    steps_per_epoch = size // hyperparameter["batch_size"] ##15827.875
    global_step = 0

    with wb.init(project="EO-SAR Change-Detection", name="cmcdnet-run-001", config=cfg) as run:
        print("####### Started Training Loop #######")
        for epoch in range(hyperparameter["epochs"]):
            start = time.perf_counter()
            model.train()
            running = 0.0
            for _, (pre, post, target, _) in enumerate(itertools.islice(train_loader, steps_per_epoch)):
                pre = pre.to(device)
                post = post.to(device)
                target = target.to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda",  enabled=torch.cuda.is_available()):
                    logits = model(pre, post)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()
                global_step += 1

                if global_step % cfg["metrics"]["log_steps"] == 0:
                    iou = iou_score(logits, target)
                    precision, recall = precision_recall(logits, target)
                    f1 = f1_score(precision, recall)

                    log = {
                        "train/loss": loss.item(),
                        "train/iou": iou,
                        "train/precision": precision,
                        "train/recall": recall,
                        "train/f1_score": f1
                    }

                    run.log(log, step=global_step)

                if cfg["metrics"]["log_images"] and global_step % cfg["metrics"]["image_log_every"] == 0:
                    pred = (torch.sigmoid(logits) > 0.5).squeeze(1)
                    log_images(pre, post, pred, target, step=global_step, max_images=2)

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

                    pred = (torch.sigmoid(logits) > 0.5)
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
                    "val/f1_score": f1
                }

                run.log(val_log, step=epoch)
            epoch_time = time.perf_counter() - start
            print(
                f"epoch {epoch + 1}/{hyperparameter['epochs']} | "
                f"train_loss {train_epoch_loss:.4f} | "
                f"val_loss {val_avg_loss:.4f} | "
                f"val_iou {val_avg_iou:.4f} | "
                f"val_f1 {f1:.4f} | "
                f"time {epoch_time:.2f}s"
            )
            run.log({"train/per_epoch_sec": epoch_time}, step=epoch)
            run.log({"train/epoch_loss": train_epoch_loss}, step=global_step)
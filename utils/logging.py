import wandb as wb
import numpy as np

def log_images(pre, post, pred_mask, gt_mask, step, max_images=2):
    # pre/post: [B, C, H, W] (float), pred/gt: [B, H, W] (int/float)
    # Convert to CPU numpy for W&B
    pre = pre[:max_images].detach().cpu()
    post = post[:max_images].detach().cpu()
    pred = pred_mask[:max_images].detach().cpu()
    gt = gt_mask[:max_images].detach().cpu()

    images = []
    class_labels = {0: "no-change", 1: "change"}

    for i in range(pre.shape[0]):
        # Make a 3‑channel visualization
        def to_3ch(x):
            x = x[0] if x.shape[0] == 1 else x[:3]  # use first 3 channels if >1
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            x = (x * 255).byte().numpy()
            return np.transpose(x, (1, 2, 0))

        pre_img = to_3ch(pre[i])
        post_img = to_3ch(post[i])

        images.append(
            wb.Image(
                np.concatenate([pre_img, post_img], axis=1),
                caption=f"pre | post | step {step}",
                masks={
                    "pred": {"mask_data": pred[i].numpy().astype(np.uint8), "class_labels": class_labels},
                    "gt": {"mask_data": gt[i].numpy().astype(np.uint8), "class_labels": class_labels},
                },
            )
        )

    wb.log({"examples": images}, step=step)
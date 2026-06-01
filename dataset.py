import rasterio
import numpy as np
import os
import typing
import yaml
from pathlib import Path

from torch.utils.data import Dataset 
from torchvision.transforms import v2
from skimage.filters import sobel
from utils.mask_utils import remap_labels

config_path = Path(__file__).resolve().parent / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

threshold = cfg["dataset_thresholds"]

class ChangeDetctionDataset(Dataset):
    def __init__(self,
                 pre_img_path: str,
                 post_img_path: str,
                 target_img_path: str,
                 patch_size: int,
                 stride: int,
                 index_path: typing.Optional[str], 
                 build_metadata: bool = True,
                 transform: typing.Optional[str] = None):
        
        self.pre_img_path = pre_img_path
        self.post_img_path = post_img_path
        self.target_img_path = target_img_path
        self.patch_size = patch_size
        self.stride = stride
        self.height = 1024
        self.width = 1024

        self.pre_files = sorted(os.listdir(self.pre_img_path))
        self.post_files = sorted(os.listdir(self.post_img_path))
        self.target_files = sorted(os.listdir(self.target_img_path))

        self.coords = self._generate_patch_coordinates()
        self.build_metadata = build_metadata
        self.transform = transform
        if self.build_metadata:
            if index_path and os.path.exists(index_path):
                self.patch_metadata = self._load_patch_metadata(index_path)
            else:
                self.patch_metadata, self.temp1, self.temp2 = self._build_patch_metadata()
                if index_path:
                    # os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    np.savez(index_path, metadata=np.array(self.patch_metadata, dtype=object))
        else:
            self.patch_metadata = None
            self.total_patches = len(self.pre_files) * len(self.coords)
    
    def _generate_patch_coordinates(self):
        coords = []
        for y in range(0, self.height - self.patch_size + 1, self.stride):
            for x in range(0, self.width - self.patch_size + 1, self.stride):
                coords.append((x, y))
        return coords

    def _integral_image(self, arr: np.ndarray) -> np.ndarray:
        return arr.cumsum(axis=0).cumsum(axis=1)

    def _window_sum(self, ii: np.ndarray, x: int, y: int, h: int, w: int) -> float:
        x2 = x + w - 1
        y2 = y + h - 1
        total = ii[y2, x2]
        if y > 0:
            total -= ii[y - 1, x2]
        if x > 0:
            total -= ii[y2, x - 1]
        if x > 0 and y > 0:
            total += ii[y - 1, x - 1]
        return float(total)

    def _normalize01(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float16)
        min_val = float(img.min())
        max_val = float(img.max())
        return (img - min_val) / (max_val - min_val + 1e-6)

    def _read_patch(self, folder_path, filename, x, y):
        file_path = os.path.join(folder_path, filename)
        with rasterio.open(file_path) as src:
            window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)
            data = src.read(window=window)
            if src.nodata is not None:
                data = np.where(data == src.nodata, 0, data)
            data = np.nan_to_num(data, nan=0.0)

        data = np.transpose(data, (1, 2, 0)).astype(np.float16)
        return data
    
    def _load_patch_metadata(self, index_path: str):
        data = np.load(index_path, allow_pickle=True)
        return data["metadata"].tolist()     
    
    def _build_patch_metadata(self):
        metadata = []
        score_list = []
        change_list = []

        for image_idx in range(len(self.pre_files)):
            with rasterio.open(os.path.join(self.pre_img_path, self.pre_files[image_idx])) as src1, rasterio.open(os.path.join(self.post_img_path, self.post_files[image_idx])) as src2, rasterio.open(os.path.join(self.target_img_path, self.target_files[image_idx])) as target:
                pre_full = src1.read(masked=True)
                target_full = target.read(masked=True)
                post_full = src2.read(masked=True)

                pre_img = np.ma.filled(pre_full, 0.0)
                post_img = np.ma.filled(post_full, 0.0)

                if pre_img.ndim == 3:
                    pre2d = pre_img.mean(axis=0) if pre_img.shape[0] > 1 else pre_img[0]
                else:
                    pre2d = pre_img

                if post_img.ndim == 3:
                    post2d = post_img.mean(axis=0) if post_img.shape[0] > 1 else post_img[0]
                else:
                    post2d = post_img

                pre2d = self._normalize01(pre2d)
                post2d = self._normalize01(post2d)

                grad_map = sobel(post2d).astype(np.float32)
                grad_ii = self._integral_image(grad_map)

                for idx, (x, y) in enumerate(self.coords):
                    pre = pre_full[:, y:y+self.patch_size, x:x+self.patch_size]
                    valid_mask = ~np.ma.getmaskarray(pre)
                    valid_mask = np.any(valid_mask, axis=0)

                    post_patch = post_img[:, y:y+self.patch_size, x:x+self.patch_size]
                    post_valid_mask = np.any(post_patch != 0, axis=0)

                    combined_valid = valid_mask & post_valid_mask
                    valid_ratio = combined_valid.mean()

                    if valid_ratio < threshold["valid_thr"]:
                        metadata.append({
                            "image_idx": image_idx,
                            "patch_idx": idx,
                            "x": x,
                            "y": y,
                            "bucket": "discard",
                        })
                        continue

                    label = target_full[:, y:y+self.patch_size, x:x+self.patch_size]
                    label = label.astype(np.uint8)
                    label = np.nan_to_num(label, 0)
                    label = np.transpose(label, (1, 2, 0))
                    label = label.squeeze()
                    binary_mask = remap_labels(label).astype(np.uint8)
                    changed_pixel = binary_mask.sum()
                    changed_ratio = (changed_pixel/max(1, combined_valid.sum()))

                    score = self._window_sum(grad_ii, x, y, self.patch_size, self.patch_size) / (self.patch_size * self.patch_size)
                    score_list.append(score)
                    change_list.append(changed_ratio)

                    if changed_ratio <= threshold["change_thr"]:
                        if score >= threshold["hard_neg_thr"]:
                            metadata.append({
                                "image_idx": image_idx,
                                "patch_idx": idx,
                                "x": x,
                                "y": y,
                                "bucket": "hard_negative",
                            })
                        else:
                            metadata.append({
                                "image_idx": image_idx,
                                "patch_idx": idx,
                                "x": x,
                                "y": y,
                                "bucket": "trivial",
                            })
                    elif score >= threshold["informative_thr"]:
                        metadata.append({
                            "image_idx": image_idx,
                            "patch_idx": idx,
                            "x": x,
                            "y": y,
                            "bucket": "informative",
                        })
                    else:
                        metadata.append({
                            "image_idx": image_idx,
                            "patch_idx": idx,
                            "x": x,
                            "y": y,
                            "bucket": "trivial",
                        })

        return metadata, score_list, change_list
                        
         
    def __len__(self):
        if self.patch_metadata is not None:
            return len(self.patch_metadata)
        return self.total_patches
    
    def __getitem__(self, idx):
        if self.build_metadata:
            image_idx = int(self.patch_metadata[idx]["image_idx"])
            patch_idx = int(self.patch_metadata[idx]["patch_idx"])
            x = self.patch_metadata[idx]["x"]
            y = self.patch_metadata[idx]["y"]
            meta = self.patch_metadata[idx]
        else:
            coord_count = len(self.coords)
            image_idx = idx // coord_count
            coord_idx = idx % coord_count
            x, y = self.coords[coord_idx]
            meta = {"image_idx": image_idx, "patch_idx": coord_idx, "x": x, "y": y}

        pre = self._read_patch(self.pre_img_path, self.pre_files[image_idx], x, y)
        post = self._read_patch(self.post_img_path, self.post_files[image_idx], x, y)
        target = self._read_patch(self.target_img_path, self.target_files[image_idx], x, y)

        target = remap_labels(target)
        if self.transform:
            transformed = self.transform(
                image1=pre,
                image2=post,
                mask=target,
            )
            pre = transformed["image1"]
            post = transformed["image2"]
            target = transformed["mask"]
        
        return pre, post, target, meta
import numpy as np
import torch
import random
import yaml

from torchvision.transforms import v2

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

aug = cfg["augmentations"]


class RandomIntensityScale:
    def __call__(self, x):
        scale = torch.empty(1).uniform_(0.98, 1.02)
        return x * scale

class PairedTransform:
    def __init__(
            self,
            horizontal_flip_p=aug["horizontal_flip"]["probability"],
            vertical_flip_p=aug["vertical_flip"]["probability"]
        ):
        # normalize: callable that expects (image1, image2) -> (image1, image2)
        # augment: callable that expects (image1, image2, mask) -> (image1, image2, mask)
        # to_tensor: callable that expects (image1, image2, mask) -> (image1, image2, mask)
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.optical_augment = v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=aug["color_jitter"]["brightness"],
                    contrast=aug["color_jitter"]["contrast"],
                    saturation=aug["color_jitter"]["saturation"],
                    hue=aug["color_jitter"]["hue"],
                ),
            ], p=aug["color_jitter"]["probability"])
        self.sar_augment = v2.RandomApply(
            [
                RandomIntensityScale(),
            ], p=aug["intensity_scale"]["probability"])
        
        self.blur = v2.RandomApply(
            [
                v2.GaussianBlur(kernel_size=3)
            ], p=aug["gaussian_blur"]["probability"])

    def __call__(self, image1, image2, mask):

        image2 = self._sar_normalization(image2)
        image1 = self._optical_normalization(image1)

        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask.squeeze()).long()

        image1 = self.optical_augment(image1)
        image2 = self.sar_augment(image2)

        image1 = self.blur(image1)
        image2 = self.blur(image2)

        image1, image2, mask = self._paired_augment(image1, image2, mask)


        return {"image1": image1, "image2": image2, "mask": mask}

    def _sar_normalization(self, image2):
        p2, p98 = np.percentile(image2, (2, 98))
        image2 = np.clip(image2, p2, p98)
        image2 = (image2 - image2.min())/(image2.max() - image2.min() + 1e-6)
        return image2
        
    def _optical_normalization(self, image1):
        p2, p98 = np.percentile(image1, (2, 98))
        image1 = np.clip(image1, p2, p98)
        image1 = (image1 - image1.min())/(image1.max() - image1.min() + 1e-6)
        return image1
    
    def _paired_augment(self, image1, image2, mask):
        if random.random() < self.horizontal_flip_p:
            image1 = v2.functional.hflip(image1)
            image2 = v2.functional.hflip(image2)
            mask = v2.functional.hflip(mask)

        if random.random() < self.vertical_flip_p:
            image1 = v2.functional.vflip(image1)
            image2 = v2.functional.vflip(image2)
            mask = v2.functional.vflip(mask)
        
        return image1, image2, mask


class ValTransform:
    def __init__(self):
        pass
    def __call__(self, image1, image2, mask):
        image2 = self._sar_normalization(image2)
        image1 = self._optical_normalization(image1)

        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)
        image2 = torch.from_numpy(image2).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask.squeeze()).long()

        return {"image1": image1, "image2": image2, "mask": mask}

    def _sar_normalization(self, image2):
        p2, p98 = np.percentile(image2, (2, 98))
        image2 = np.clip(image2, p2, p98)
        image2 = (image2 - image2.min())/(image2.max() - image2.min() + 1e-6)
        return image2
        
    def _optical_normalization(self, image1):
        p2, p98 = np.percentile(image1, (2, 98))
        image1 = np.clip(image1, p2, p98)
        image1 = (image1 - image1.min())/(image1.max() - image1.min() + 1e-6)
        return image1

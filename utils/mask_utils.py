import numpy as np

LABEL_MAP = {
    0: 0,  # Background -> No Change
    1: 0,  # Intact -> No Change
    2: 1,  # Damaged -> Change
    3: 1   # Destroyed -> Change
}


def remap_labels(mask: np.ndarray) -> np.ndarray:

    remapped = np.zeros_like(mask, dtype=np.uint8)
    remapped[(mask == 2) | (mask == 3)] = 1
    return remapped.astype(np.uint8)
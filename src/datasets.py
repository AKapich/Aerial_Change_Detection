import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DatasetLEVIR(Dataset):
    def __init__(
        self, root_dir: str, split: str = "train", patch_size: int = 256, transform=None
    ):
        """
        Args:
            root_dir: path to dataset folder
            split: 'train', 'val', or 'test'
            patch_size: size to crop full 1024x1024 images into
            transform: albumentations transform
        """
        self.root = Path(root_dir) / split
        self.patch_size = patch_size
        self.transform = transform
        self.filenames = sorted(os.listdir(self.root / "A"))

        self.patches = []
        img_size = 1024
        step = patch_size
        for fname in self.filenames:
            for y in range(0, img_size, step):
                for x in range(0, img_size, step):
                    self.patches.append((fname, y, x))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        fname, y, x = self.patches[idx]
        p = self.patch_size

        img_A = cv2.imread(str(self.root / "A" / fname))
        img_B = cv2.imread(str(self.root / "B" / fname))
        mask = cv2.imread(str(self.root / "label" / fname), cv2.IMREAD_GRAYSCALE)

        img_A = img_A[y : y + p, x : x + p]
        img_B = img_B[y : y + p, x : x + p]
        mask = mask[y : y + p, x : x + p]

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # binarizing mask (LEVIR labels are 0 or 255)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            spatial_transform, color_transform = self.transform

            augmented = spatial_transform(image=img_A, image0=img_B, mask=mask)
            img_A = augmented["image"]
            img_B = augmented["image0"]
            mask = augmented["mask"]

            if color_transform is not None:
                img_A = color_transform(image=img_A)["image"]
                img_B = color_transform(image=img_B)["image"]

        img_A = torch.FloatTensor(img_A).permute(2, 0, 1) / 255.0
        img_B = torch.FloatTensor(img_B).permute(2, 0, 1) / 255.0
        image = torch.cat([img_A, img_B], dim=0)

        mask = torch.LongTensor(mask)

        return image, mask

import torch
from torch.utils.data import DataLoader
import numpy as np


def collect_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """
    Returns:
        all_probs:  (N,) numpy float array — probability of change per pixel
        all_labels: (N,) numpy int array   — ground truth (0 or 1)
    """
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, 1]  # (B, H, W)
            all_probs.append(probs.cpu().numpy().ravel())
            all_labels.append(masks.numpy().ravel())
    return np.concatenate(all_probs), np.concatenate(all_labels)

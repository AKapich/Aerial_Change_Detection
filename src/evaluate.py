import torch
import numpy as np
import yaml
import argparse
from torch.utils.data import DataLoader

from datasets import DatasetLEVIR
from model import build_model
from transforms import get_val_transforms
from metrics import MetricAccumulator
from visualize import save_prediction_grid
from utils import collect_probs
from postprocess import filter_small_components


def evaluate(
    config_path: str,
    checkpoint_path: str,
    save_visualizations: bool = False,
    threshold: float = 0.5,
    min_component_pixels: int = 0,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = DatasetLEVIR(
        cfg["data"]["root_dir"],
        split="test",
        patch_size=cfg["training"]["patch_size"],
        transform=get_val_transforms(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(
        f"\nEvaluating with threshold = {threshold:.4f}"
        + (
            f", min_component_pixels = {min_component_pixels}"
            if min_component_pixels > 0
            else ""
        )
    )
    test_metrics = MetricAccumulator()

    if min_component_pixels > 0:
        # per-patch loop to preserve spatial 2D structure needed for component filtering.
        with torch.no_grad():
            for images, masks in test_loader:
                probs = (
                    torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
                )  # (B, H, W)
                for i in range(probs.shape[0]):
                    pred = (probs[i] >= threshold).astype(np.uint8)
                    pred = filter_small_components(
                        pred, min_pixels=min_component_pixels
                    )
                    test_metrics.update(
                        torch.from_numpy(pred.astype(np.int64)), masks[i]
                    )
    else:
        all_probs, all_labels = collect_probs(model, test_loader, device)
        test_metrics.update(
            torch.from_numpy(all_probs),
            torch.from_numpy(all_labels),
            threshold=threshold,
        )

    m = test_metrics.compute()

    print(f"Change IoU:  {m['iou_change']:.4f}")
    print(f"F1 (change): {m['f1']:.4f}")
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall:      {m['recall']:.4f}")

    if save_visualizations:
        save_prediction_grid(test_ds, model, device, n_samples=8)

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        save_visualizations=args.save_visualizations,
        threshold=args.threshold,
        min_component_pixels=args.min_component_pixels,
    )

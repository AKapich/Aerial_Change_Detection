import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
from utils import collect_probs


def find_optimal_threshold(
    all_probs: np.ndarray, all_labels: np.ndarray, save_plot: bool = True
) -> float:
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = f1_scores[best_idx]
    default_f1 = f1_scores[np.argmin(np.abs(thresholds - 0.5))]

    print(f"Default threshold (0.50): F1 = {default_f1:.4f}")
    print(f"Optimal threshold ({best_threshold:.3f}): F1 = {best_f1:.4f}")
    print(f"Improvement: +{best_f1 - default_f1:.4f}")

    if save_plot:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, lw=2)
        plt.scatter(
            recall[best_idx],
            precision[best_idx],
            color="red",
            zorder=5,
            label=f"Best threshold={best_threshold:.2f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(thresholds, f1_scores[:-1], lw=2)
        plt.axvline(
            best_threshold,
            color="red",
            linestyle="--",
            label=f"Best={best_threshold:.2f}",
        )
        plt.axvline(0.5, color="gray", linestyle="--", label="Default=0.50")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 vs Threshold")
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/threshold_analysis.png", dpi=150, bbox_inches="tight")
        print("Saved results/threshold_analysis.png")

    return best_threshold


if __name__ == "__main__":
    import argparse
    import sys
    import os
    import yaml

    sys.path.insert(0, os.path.dirname(__file__))
    from datasets import DatasetLEVIR
    from model import build_model
    from transforms import get_val_transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = DatasetLEVIR(
        cfg["data"]["root_dir"],
        split=args.split,
        patch_size=cfg["training"]["patch_size"],
        transform=get_val_transforms(),
    )
    loader = DataLoader(
        ds,
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print(f"Collecting probabilities from '{args.split}' split")
    all_probs, all_labels = collect_probs(model, loader, device)

    best_threshold = find_optimal_threshold(
        all_probs, all_labels, save_plot=not args.no_plot
    )
    print(f"\nUse threshold {best_threshold:.4f}")

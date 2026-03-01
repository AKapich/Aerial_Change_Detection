import torch
import yaml
import argparse
from torch.utils.data import DataLoader

from datasets import DatasetLEVIR
from model import build_model
from transforms import get_val_transforms
from metrics import MetricAccumulator
from visualize import save_prediction_grid


def evaluate(config_path: str, checkpoint_path: str, save_visualizations: bool = False):
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

    test_metrics = MetricAccumulator()

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_metrics.update(preds.cpu(), masks.cpu())

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
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, save_visualizations=args.save_visualizations)

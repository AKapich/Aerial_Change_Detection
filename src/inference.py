import argparse
import yaml
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import build_model
from transforms import get_val_transforms
from postprocess import filter_small_components


def predict(
    img_A_path: str,
    img_B_path: str,
    config_path: str,
    checkpoint_path: str,
    output_path: str = "results/prediction.png",
    threshold: float = 0.5,
    min_component_pixels: int = 50,
):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    patch_size = cfg["training"]["patch_size"]
    encoder_name = cfg["model"]["encoder_name"]
    encoder_weights = cfg["model"].get("encoder_weights", None)
    num_classes = cfg["model"]["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    val_transform, _ = get_val_transforms()

    def load_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    img_A_full = load_image(img_A_path)
    img_B_full = load_image(img_B_path)

    H, W = img_A_full.shape[:2]
    pred_full = np.zeros((H, W), dtype=np.int64)

    with torch.no_grad():
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                patch_A = img_A_full[y : y + patch_size, x : x + patch_size]
                patch_B = img_B_full[y : y + patch_size, x : x + patch_size]

                if val_transform:
                    augmented = val_transform(image=patch_A, image0=patch_B)
                    patch_A = augmented["image"]
                    patch_B = augmented["image0"]

                t_A = torch.FloatTensor(patch_A).permute(2, 0, 1) / 255.0
                t_B = torch.FloatTensor(patch_B).permute(2, 0, 1) / 255.0
                combined = torch.cat([t_A, t_B], dim=0).unsqueeze(0).to(device)

                change_prob = torch.softmax(model(combined), dim=1)[0, 1].cpu().numpy()
                patch_pred = (change_prob >= threshold).astype(np.uint8)
                pred_full[y : y + patch_size, x : x + patch_size] = patch_pred

    if min_component_pixels > 0:
        pred_full = filter_small_components(
            pred_full.astype(np.uint8), min_pixels=min_component_pixels
        )

    img_A_disp = img_A_full
    img_B_disp = img_B_full
    pred = pred_full

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_A_disp)
    axes[0].set_title("Before")
    axes[1].imshow(img_B_disp)
    axes[1].set_title("After")
    axes[2].imshow(pred, cmap="Reds")
    axes[2].set_title("Detected Changes")
    for ax in axes:
        ax.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_A", required=True)
    parser.add_argument("--img_B", required=True)
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="results/prediction.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    predict(
        args.img_A,
        args.img_B,
        args.config,
        args.checkpoint,
        args.output,
        threshold=args.threshold,
        min_component_pixels=args.min_component_pixels,
    )

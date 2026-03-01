import yaml
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

from datasets import DatasetLEVIR
from model import build_model
from transforms import get_train_transforms, get_val_transforms
from metrics import MetricAccumulator


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = DatasetLEVIR(
        cfg["data"]["root_dir"],
        split="train",
        patch_size=cfg["training"]["patch_size"],
        transform=get_train_transforms(),
    )
    val_ds = DatasetLEVIR(
        cfg["data"]["root_dir"],
        split="val",
        patch_size=cfg["training"]["patch_size"],
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )

    model = build_model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    change_weight = cfg["training"][
        "class_weight_change"
    ]  # change pixel imbalance mitigation
    class_weights = torch.FloatTensor([1.0, change_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"]
    )

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.log_params(
            {
                "encoder": cfg["model"]["encoder_name"],
                "batch_size": cfg["training"]["batch_size"],
                "lr": cfg["training"]["learning_rate"],
                "epochs": cfg["training"]["num_epochs"],
                "patch_size": cfg["training"]["patch_size"],
                "loss": cfg["training"]["loss"],
                "optimizer": cfg["training"]["optimizer"],
                "scheduler": cfg["training"]["scheduler"],
                "class_weight_change": cfg["training"]["class_weight_change"],
            }
        )

        best_val_iou = 0.0
        best_model_path = "checkpoints/best_model.pth"
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(cfg["training"]["num_epochs"]):
            model.train()
            train_loss = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)  # (N, 2, H, W)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_metrics = MetricAccumulator()

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    val_metrics.update(preds.cpu(), masks.cpu())

            val_loss /= len(val_loader)
            metrics = val_metrics.compute()

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_change_iou": metrics["iou_change"],
                    "val_f1": metrics["f1"],
                    "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"],
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_iou={metrics['iou_change']:.4f}, val_f1={metrics['f1']:.4f}"
            )

            if metrics["iou_change"] > best_val_iou:
                best_val_iou = metrics["iou_change"]
                torch.save(model.state_dict(), best_model_path)
                print(f"new best: {best_val_iou:.4f}")

        mlflow.log_metric("best_val_change_iou", best_val_iou)
        mlflow.log_artifact(best_model_path)
        print(f"\nTraining complete. Best change IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    args = parser.parse_args()
    train(args.config)

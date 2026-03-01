import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_prediction_grid(
    dataset, model, device, n_samples=8, output_path="results/prediction_grid.png"
):
    os.makedirs("results", exist_ok=True)

    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples * 4))
    axes[0][0].set_title("Before", fontsize=12)
    axes[0][1].set_title("After", fontsize=12)
    axes[0][2].set_title("Ground Truth", fontsize=12)
    axes[0][3].set_title("Prediction", fontsize=12)

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu()

            img_A = image[:3].permute(1, 2, 0).numpy()
            img_B = image[3:].permute(1, 2, 0).numpy()

            axes[row][0].imshow(img_A)
            axes[row][1].imshow(img_B)
            axes[row][2].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row][3].imshow(pred.numpy(), cmap="gray", vmin=0, vmax=1)

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

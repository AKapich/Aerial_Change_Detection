import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
) -> nn.Module:
    """
    UNet w/ pretrained ResNet backbone.
    Input: 6-channel image (3 before + 3 after)
    Output: raw logits [B, num_classes, H, W]
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=6,
        classes=num_classes,
    )  # activation handled within cross-entropy loss
    return model

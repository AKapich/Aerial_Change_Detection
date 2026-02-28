import albumentations as A


def get_train_transforms():
    spatial = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"image0": "image"},
    )

    color = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3)

    return spatial, color


def get_val_transforms():
    spatial = A.Compose([], additional_targets={"image0": "image"})
    return spatial, None

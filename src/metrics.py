import torch


class MetricAccumulator:
    """Tracking TP/FP/FN across batches for epoch-level metric computation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.union = 0.0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        threshold: float | None = None,
    ):
        if threshold is not None:
            pred_change = preds >= threshold
        else:
            pred_change = preds == 1
        target_change = targets == 1

        self.tp += (pred_change & target_change).sum().item()
        self.fp += (pred_change & ~target_change).sum().item()
        self.fn += (~pred_change & target_change).sum().item()
        self.union += (pred_change | target_change).sum().item()

    def compute(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        iou = self.tp / self.union if self.union > 0 else 0.0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou_change": iou,
        }

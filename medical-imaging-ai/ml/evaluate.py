"""Evaluation script computing classification AUROC/F1 and 3D segmentation Dice score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def compute_dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Compute binary Dice Similarity Coefficient between predicted and ground-truth masks."""
    pred = (pred_mask > 0.5).astype(np.float32)
    true = (true_mask > 0.5).astype(np.float32)
    intersection = float((pred * true).sum())
    return (2.0 * intersection + eps) / (float(pred.sum() + true.sum()) + eps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models and write metrics JSON")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--cls-model", default="models/classifier_densenet121.pth")
    parser.add_argument("--seg-model", default="models/unet3d_monai.pth")
    parser.add_argument("--output", default="reports/metrics.json")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "disclaimer": "FOR EDUCATION ONLY - NOT FOR CLINICAL USE",
        "status": "evaluation_template_ready",
        "notes": "Populate with real hold-out metrics after training on CheXpert / NIH ChestX-ray14 and MSD.",
        "metrics": {
            "classification_auroc_macro": None,
            "segmentation_dice_mean": None,
        },
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote evaluation report to {out}")


if __name__ == "__main__":
    main()

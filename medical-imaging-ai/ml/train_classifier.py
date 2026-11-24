"""Milestone 3A: 2D Chest X-ray Multi-Label Classifier (DenseNet-121 / EfficientNet-B0)."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def build_classifier(backbone: str = "densenet121", num_classes: int = 5, pretrained: bool = True):
    """Create a transfer-learning PyTorch classifier for CheXpert / NIH ChestX-ray14."""
    import torch.nn as nn
    from torchvision import models

    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def train(params_path: str, data_dir: str, output_path: str) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    with open(params_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["classifier"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        backbone=cfg["backbone"],
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", True),
    ).to(device)

    # Load preprocessed .npy tensors if present; fall back to smoke-test batch
    npy_files = sorted(Path(data_dir).glob("*.npy"))
    if npy_files:
        x_np = np.stack([np.load(p) for p in npy_files], axis=0)
        y_np = np.zeros((len(npy_files), cfg["num_classes"]), dtype=np.float32)
    else:
        x_np = np.random.randn(8, 3, 224, 224).astype(np.float32)
        y_np = (np.random.rand(8, cfg["num_classes"]) > 0.7).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np)),
        batch_size=min(cfg["batch_size"], len(x_np)),
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    model.train()
    for epoch in range(1, cfg["epochs"] + 1):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        print(f"[Classifier] Epoch {epoch}/{cfg['epochs']} - BCE loss: {running_loss / len(loader):.4f}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": cfg["backbone"], "state_dict": model.state_dict()}, out)
    print(f"Saved classifier checkpoint to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2D chest X-ray classifier")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", default="models/classifier_densenet121.pth")
    args = parser.parse_args()
    train(args.params, args.data, args.output)


if __name__ == "__main__":
    main()

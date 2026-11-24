"""Milestone 2: Preprocessing pipeline (resize, normalize, augment) tracked by DVC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


def resize_and_normalize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Resize a grayscale or RGB scan to target_size and apply channel-wise standardization."""
    arr = np.clip(image.astype(np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
    else:
        pil_img = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")

    resized = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean_arr) / std_arr
    # Return CHW float32 tensor layout
    return np.transpose(normalized, (2, 0, 1))


def augment_scan(
    tensor_chw: np.ndarray, horizontal_flip: bool = True, seed: int | None = None
) -> np.ndarray:
    """Apply lightweight deterministic-capable augmentation (horizontal flip + mild contrast)."""
    rng = np.random.default_rng(seed)
    out = tensor_chw.copy()
    if horizontal_flip and rng.random() > 0.5:
        out = out[:, :, ::-1].copy()
    scale = float(rng.uniform(0.92, 1.08))
    return out * scale


def run_preprocessing(input_dir: Path, output_dir: Path, target_size: Tuple[int, int]) -> dict:
    """Process all PNG/JPG scans in input_dir and save standardized .npy tensors to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(input_dir.glob(ext)):
            img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            tensor = resize_and_normalize(img, target_size=target_size)
            np.save(output_dir / f"{path.stem}.npy", tensor)
            processed += 1

    manifest = {"processed_count": processed, "target_size": list(target_size)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess medical imaging dataset")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    target_size = (224, 224)
    try:
        import yaml
        with open(args.params, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            size_cfg = cfg.get("dataset", {}).get("image_size_2d", [224, 224])
            target_size = (int(size_cfg[0]), int(size_cfg[1]))
    except Exception:
        pass

    manifest = run_preprocessing(Path(args.input), Path(args.output), target_size)
    print(f"Preprocessed {manifest['processed_count']} scans -> {args.output}")


if __name__ == "__main__":
    main()

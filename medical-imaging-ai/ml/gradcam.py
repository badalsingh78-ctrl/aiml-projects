"""Milestone 4: Grad-CAM visual explanation engine (Selvaraju et al., 2017)."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class GradCAM:
    """Gradient-weighted Class Activation Mapping for PyTorch CNNs (e.g., DenseNet-121)."""

    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx: int) -> np.ndarray:
        """Compute a normalized [0, 1] 2D Grad-CAM heatmap for class_idx."""
        import torch
        import torch.nn.functional as F

        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        return (cam_np - cam_min) / max(cam_max - cam_min, 1e-6)

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def apply_thermal_colormap(heatmap_01: np.ndarray) -> np.ndarray:
    """Map a [0, 1] scalar heatmap to an RGB uint8 thermal map (blue -> cyan -> yellow -> red)."""
    h = np.clip(heatmap_01.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def overlay_heatmap_on_scan(
    scan_01: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.45
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend a [0, 1] Grad-CAM heatmap onto a [0, 1] grayscale radiograph."""
    base = np.clip(scan_01.astype(np.float32), 0.0, 1.0)
    if base.ndim == 2:
        base_rgb = np.stack([base, base, base], axis=-1) * 255.0
    else:
        base_rgb = base * 255.0

    heat_rgb = apply_thermal_colormap(heatmap_01).astype(np.float32)
    blended = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8), heat_rgb.astype(np.uint8)

"""DICOM parsing, windowing, and synthetic sample generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


@dataclass
class DicomMetadata:
    patient_id: str
    study_uid: str
    modality: str
    body_part: str
    rows: int
    cols: int
    window_center: float
    window_width: float


def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float = -600.0,
    window_width: float = 1500.0,
) -> np.ndarray:
    """Apply radiology window/level clipping and scale intensities to [0, 1]."""
    arr = pixel_array.astype(np.float32)
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    clipped = np.clip(arr, lower, upper)
    denom = max(upper - lower, 1e-6)
    return (clipped - lower) / denom


def load_dicom_image(dicom_path: str | Path) -> Tuple[np.ndarray, DicomMetadata]:
    """Load a DICOM file via pydicom, apply RescaleSlope/Intercept and windowing."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required to parse .dcm files. Install via `pip install pydicom`."
        ) from exc

    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])

    if wc is not None and ww is not None:
        norm = apply_windowing(pixels, float(wc), float(ww))
    else:
        pmin, pmax = float(pixels.min()), float(pixels.max())
        norm = (pixels - pmin) / max(pmax - pmin, 1e-6)
        wc, ww = (pmin + pmax) / 2.0, max(pmax - pmin, 1.0)

    meta = DicomMetadata(
        patient_id=str(getattr(ds, "PatientID", "ANON")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "UNKNOWN")),
        modality=str(getattr(ds, "Modality", "CR")),
        body_part=str(getattr(ds, "BodyPartExamined", "CHEST")),
        rows=int(pixels.shape[0]),
        cols=int(pixels.shape[1]),
        window_center=float(wc),
        window_width=float(ww),
    )
    return norm, meta


def generate_synthetic_chest_scan(
    size: int = 224, seed: int = 42, anomaly: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate a deterministic synthetic radiograph array for unit tests & offline demos."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ny = (yy - size / 2.0) / (size / 2.0)
    nx = (xx - size / 2.0) / (size / 2.0)

    # Lung fields (two dark ovals) and central mediastinum
    left_lung = ((nx + 0.32) / 0.28) ** 2 + (ny / 0.55) ** 2
    right_lung = ((nx - 0.32) / 0.28) ** 2 + (ny / 0.55) ** 2
    img = np.full((size, size), 0.72, dtype=np.float32)
    img[left_lung < 1.0] = 0.22
    img[right_lung < 1.0] = 0.22

    # Spine & rib striations
    spine = np.exp(-((nx / 0.08) ** 2)) * 0.20
    ribs = 0.04 * np.sin(ny * 22.0) * ((left_lung < 1.0) | (right_lung < 1.0))
    img = img + spine + ribs

    focal_center = (0.35, 0.18)
    if anomaly:
        nodule = np.exp(
            -(((nx - focal_center[0]) / 0.14) ** 2 + ((ny - focal_center[1]) / 0.14) ** 2)
        )
        img = img + 0.38 * nodule

    noise = rng.normal(0.0, 0.02, size=(size, size)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return img, {"focal_x": focal_center[0], "focal_y": focal_center[1], "anomaly": float(anomaly)}

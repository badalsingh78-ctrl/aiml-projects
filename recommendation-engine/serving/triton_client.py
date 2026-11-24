"""
Milestone 7: NVIDIA Triton Inference Server Client.

Calls Triton's KServe V2 HTTP inference protocol (`/v2/models/{model}/infer`) for
`user_tower`, `dcn_ranker`, and `wide_deep_ranker`. Gracefully falls back to the local
in-process model weights when Triton Server is not running (e.g., laptop dev mode).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

import numpy as np


class TritonServingClient:
    """Lightweight KServe V2 HTTP client for Triton Inference Server."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 0.6):
        self.base_url = (base_url or os.getenv("TRITON_URL", "http://localhost:8001")).rstrip("/")
        self.timeout = timeout
        self.triton_live = self.check_health()

    def check_health(self) -> bool:
        if os.getenv("DISABLE_TRITON", "1") == "1":
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/v2/health/ready", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def infer_ranker(self, model_name: str, feature_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Send dense feature matrix to Triton ranker model (`dcn_ranker` or `wide_deep_ranker`)."""
        if not self.triton_live:
            return None
        payload = {
            "inputs": [{
                "name": "dense_features",
                "shape": list(feature_matrix.shape),
                "datatype": "FP32",
                "data": feature_matrix.astype(np.float32).ravel().tolist(),
            }]
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v2/models/{model_name}/infer",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return np.array(body["outputs"][0]["data"], dtype=np.float64)
        except Exception:
            return None

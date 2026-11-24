"""Automated unit tests for AIML-02 preprocessing, Grad-CAM, audit DB, FastAPI, and Orthanc poller."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from ml.dicom_utils import apply_windowing, generate_synthetic_chest_scan
from ml.preprocess import resize_and_normalize, augment_scan
from ml.gradcam import apply_thermal_colormap, overlay_heatmap_on_scan
from ml.evaluate import compute_dice_score
from backend.database import AuditDatabase
from backend.main import create_app
from backend.orthanc_poller import OrthancPoller


class TestAIML02Pipeline(unittest.TestCase):
    def test_windowing_and_synthetic_scan(self):
        arr = np.array([[-1500.0, -600.0, 400.0]], dtype=np.float32)
        norm = apply_windowing(arr, window_center=-600.0, window_width=1500.0)
        self.assertAlmostEqual(float(norm[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(norm[0, 1]), 0.5, places=4)
        self.assertAlmostEqual(float(norm[0, 2]), 1.0, places=4)

        scan, meta = generate_synthetic_chest_scan(size=128, seed=7, anomaly=True)
        self.assertEqual(scan.shape, (128, 128))
        self.assertGreaterEqual(float(scan.min()), 0.0)
        self.assertLessEqual(float(scan.max()), 1.0)
        self.assertEqual(meta["anomaly"], 1.0)

    def test_preprocessing_and_augmentation(self):
        scan, _ = generate_synthetic_chest_scan(size=128, seed=1)
        tensor = resize_and_normalize(scan, target_size=(224, 224))
        self.assertEqual(tensor.shape, (3, 224, 224))
        aug = augment_scan(tensor, horizontal_flip=True, seed=123)
        self.assertEqual(aug.shape, (3, 224, 224))

    def test_gradcam_colormap_and_dice(self):
        scan, _ = generate_synthetic_chest_scan(size=64, seed=3)
        heatmap = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        cmap = apply_thermal_colormap(heatmap)
        self.assertEqual(cmap.shape, (64, 64, 3))
        blended, heat_rgb = overlay_heatmap_on_scan(scan, heatmap, alpha=0.45)
        self.assertEqual(blended.shape, (64, 64, 3))
        self.assertEqual(heat_rgb.shape, (64, 64, 3))

        mask = np.zeros((16, 16), dtype=np.float32)
        mask[4:12, 4:12] = 1.0
        self.assertAlmostEqual(compute_dice_score(mask, mask), 1.0, places=5)

    def test_fastapi_upload_review_and_audit_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_audit.db"
            client = TestClient(create_app(db_path=str(db_path)))

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertIn("FOR EDUCATION ONLY", health.json()["disclaimer"])

            # Upload a synthetic PNG scan
            scan, _ = generate_synthetic_chest_scan(size=128, seed=9)
            buf = io.BytesIO()
            Image.fromarray((scan * 255).astype(np.uint8)).save(buf, format="PNG")
            buf.seek(0)
            up = client.post(
                "/studies/upload",
                data={"study_id": "STUDY-TEST-99", "patient_id": "PT-99", "modality": "CR"},
                files={"file": ("scan.png", buf, "image/png")},
            )
            self.assertEqual(up.status_code, 200)
            payload = up.json()
            self.assertEqual(payload["study_id"], "STUDY-TEST-99")
            self.assertGreater(len(payload["predictions"]), 0)

            # Radiologist approves study
            rev = client.post(
                "/studies/STUDY-TEST-99/review",
                json={"radiologist_id": "DR-TEST", "decision": "APPROVED", "notes": "Consistent with opacity"},
            )
            self.assertEqual(rev.status_code, 200)
            self.assertEqual(rev.json()["audit_entry"]["decision"], "APPROVED")

            # Verify audit log persistence
            logs = client.get("/audit-log").json()["logs"]
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["study_id"], "STUDY-TEST-99")
            self.assertEqual(logs[0]["radiologist_id"], "DR-TEST")

    def test_orthanc_poller_stable_study_detection(self):
        poller = OrthancPoller()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "Last": 14,
            "Changes": [
                {"ChangeType": "NewInstance", "ID": "inst-1"},
                {"ChangeType": "StableStudy", "ID": "study-abc"},
            ],
        }
        mock_session.get.return_value = mock_resp

        studies = poller.fetch_new_stable_studies(session=mock_session)
        self.assertEqual(studies, ["study-abc"])
        self.assertEqual(poller.last_seq, 14)


    def test_docker_nginx_and_colab_notebook(self):
        root = Path(__file__).resolve().parents[1]
        nginx = (root / "docker" / "nginx.conf").read_text(encoding="utf-8")
        self.assertIn("proxy_pass http://backend:8000/;", nginx)
        import json
        nb = json.loads((root / "notebooks" / "AIML02_Train_Models_GPU.ipynb").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(nb["cells"]), 4)


    def test_diagnostic_report_auto_modality_and_reanalyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_reanalyze.db"
            client = TestClient(create_app(db_path=str(db_path)))

            scan, _ = generate_synthetic_chest_scan(size=128, seed=11)
            buf = io.BytesIO()
            Image.fromarray((scan * 255).astype(np.uint8)).save(buf, format="PNG")
            buf.seek(0)

            # Upload with MRI filename cue + AUTO modality
            up = client.post(
                "/studies/upload",
                data={"study_id": "STUDY-MRI-TEST", "modality": "AUTO", "body_part": "AUTO"},
                files={"file": ("test_05_brain_mri_flair_tumor.png", buf, "image/png")},
            )
            self.assertEqual(up.status_code, 200)
            payload = up.json()
            self.assertEqual(payload["modality"], "MRI")
            self.assertIn("summary", payload["report"])
            self.assertIn("headline", payload["report"])
            self.assertGreater(len(payload["report"]["recommendations"]), 0)

            # Re-run AI with CLAHE + Sharpen + Denoise enhancements
            re_resp = client.post(
                "/studies/STUDY-MRI-TEST/reanalyze",
                json={"clahe": True, "sharpen": True, "denoise": True, "contrast": 1.25},
            )
            self.assertEqual(re_resp.status_code, 200)
            re_data = re_resp.json()
            self.assertGreaterEqual(len(re_data["enhancements_applied"]), 3)
            self.assertIn("study", re_data)


if __name__ == "__main__":
    unittest.main()

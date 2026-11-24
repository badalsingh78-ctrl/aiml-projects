"""FastAPI backend for AI-Assisted Medical Imaging Prototype."""

from __future__ import annotations

import io
import os
import uuid
from typing import Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database import AuditDatabase
from backend.inference_service import (
    MedicalInferenceEngine,
    decode_b64_to_scan01,
    detect_modality_and_body_part,
    enhance_scan,
)
from ml.dicom_utils import generate_synthetic_chest_scan


DISCLAIMER = "FOR EDUCATION ONLY - NOT FOR CLINICAL USE"


class ReviewRequest(BaseModel):
    radiologist_id: str
    decision: str
    notes: str = ""


class ReanalyzeRequest(BaseModel):
    modality: Optional[str] = None
    clahe: bool = False
    sharpen: bool = False
    denoise: bool = False
    invert: bool = False
    contrast: float = 1.0
    brightness: float = 0.0
    heatmap_threshold: float = 0.0


def create_app(db_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(
        title="AI-Assisted Medical Imaging Prototype API",
        version="2.0.0",
        description="FastAPI inference, Grad-CAM explainability, image enhancement, and radiologist audit API.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db = AuditDatabase(db_path or os.getenv("DB_PATH", "data/audit_log.db"))
    engine = MedicalInferenceEngine(os.getenv("CLASSIFIER_WEIGHTS"))

    if not db.list_studies():
        scan, _ = generate_synthetic_chest_scan(size=224, seed=42, anomaly=True)
        res = engine.infer_2d_scan(scan, modality="CR", filename="demo_chest_effusion.png")
        db.upsert_study(
            {
                "study_id": "STUDY-DEMO-001",
                "patient_id": "SYNTH-PT-001",
                "modality": "CR",
                "body_part": "CHEST PA",
                "source": "SYNTHETIC_DEMO",
                "raw_scan_b64": res["scan_b64"],
                **res,
            }
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "engine_mode": engine.mode, "disclaimer": DISCLAIMER}

    @app.get("/studies")
    def list_studies():
        return {"studies": db.list_studies()}

    @app.get("/studies/{study_id}")
    def get_study(study_id: str):
        study = db.get_study(study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")
        return {"study": study}

    @app.post("/studies/upload")
    async def upload_study(
        file: UploadFile = File(...),
        study_id: Optional[str] = Form(None),
        patient_id: str = Form("ANON-PATIENT"),
        modality: str = Form("AUTO"),
        body_part: str = Form("AUTO"),
        source: str = Form("MANUAL_UPLOAD"),
    ):
        raw = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(raw)).convert("L").resize((224, 224))
            scan_01 = np.asarray(pil_img, dtype=np.float32) / 255.0
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        detected_mod, detected_part = detect_modality_and_body_part(file.filename or "", scan_01)
        final_mod = detected_mod if modality.upper() == "AUTO" else modality.upper()
        final_part = detected_part if body_part.upper() == "AUTO" else body_part

        sid = study_id or f"STUDY-{uuid.uuid4().hex[:8].upper()}"
        result = engine.infer_2d_scan(scan_01, modality=final_mod, filename=file.filename or "")
        record = {
            "study_id": sid,
            "patient_id": patient_id,
            "modality": final_mod,
            "body_part": final_part,
            "source": source,
            "status": "PENDING_REVIEW",
            "raw_scan_b64": result["scan_b64"],
            **result,
        }
        db.upsert_study(record)
        return record

    @app.post("/studies/{study_id}/reanalyze")
    def reanalyze_study(study_id: str, req: ReanalyzeRequest):
        study = db.get_study(study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")

        base_b64 = study.get("raw_scan_b64") or study.get("scan_b64")
        if not base_b64:
            raise HTTPException(status_code=400, detail="Study has no image payload")

        prev_conf = float(study.get("confidence") or 0.0)
        scan_01 = decode_b64_to_scan01(base_b64, size=(224, 224))
        enhanced_01, applied_tools = enhance_scan(
            scan_01,
            clahe=req.clahe,
            sharpen=req.sharpen,
            denoise=req.denoise,
            invert=req.invert,
            contrast=req.contrast,
            brightness=req.brightness,
        )

        mod = (req.modality or study.get("modality") or "CR").upper()
        body_part = study.get("body_part") or "CHEST"
        if req.modality and req.modality.upper() != study.get("modality"):
            body_part = {"CR": "CHEST PA", "CT": "CHEST CT", "MRI": "BRAIN MRI"}.get(mod, body_part)

        res = engine.infer_2d_scan(
            enhanced_01,
            modality=mod,
            enhancements=applied_tools,
            heatmap_threshold=req.heatmap_threshold,
        )
        updated = {
            **study,
            "modality": mod,
            "body_part": body_part,
            "raw_scan_b64": base_b64,
            **res,
        }
        db.upsert_study(updated)
        return {
            "study": updated,
            "previous_confidence": round(prev_conf, 4),
            "confidence_delta": round(float(res["confidence"]) - prev_conf, 4),
            "enhancements_applied": applied_tools,
        }

    @app.post("/studies/{study_id}/review")
    def review_study(study_id: str, req: ReviewRequest):
        try:
            entry = db.record_decision(
                study_id=study_id,
                radiologist_id=req.radiologist_id,
                decision=req.decision,
                notes=req.notes,
            )
            return {"audit_entry": entry}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/audit-log")
    def audit_log():
        return {"logs": db.list_audit_logs()}

    return app


app = create_app()

"""SQLite persistence for studies, AI diagnostic reports, and radiologist audit logs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditDatabase:
    def __init__(self, db_path: str | Path = "data/audit_log.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    body_part TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
                    top_finding TEXT,
                    confidence REAL,
                    predictions_json TEXT,
                    report_json TEXT,
                    raw_scan_b64 TEXT,
                    scan_b64 TEXT,
                    heatmap_b64 TEXT,
                    overlay_b64 TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT NOT NULL,
                    radiologist_id TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK (decision IN ('APPROVED', 'REJECTED')),
                    ai_finding TEXT NOT NULL,
                    ai_confidence REAL NOT NULL,
                    notes TEXT,
                    decided_at TEXT NOT NULL,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id)
                );
                """
            )
            # Migrate older databases smoothly without losing existing studies
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(studies)").fetchall()
            }
            if "report_json" not in existing_cols:
                conn.execute("ALTER TABLE studies ADD COLUMN report_json TEXT")
            if "raw_scan_b64" not in existing_cols:
                conn.execute("ALTER TABLE studies ADD COLUMN raw_scan_b64 TEXT")

    def upsert_study(self, study: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO studies (
                    study_id, patient_id, modality, body_part, source, status,
                    top_finding, confidence, predictions_json, report_json,
                    raw_scan_b64, scan_b64, heatmap_b64, overlay_b64, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(study_id) DO UPDATE SET
                    modality=excluded.modality,
                    body_part=excluded.body_part,
                    top_finding=excluded.top_finding,
                    confidence=excluded.confidence,
                    predictions_json=excluded.predictions_json,
                    report_json=excluded.report_json,
                    scan_b64=excluded.scan_b64,
                    heatmap_b64=excluded.heatmap_b64,
                    overlay_b64=excluded.overlay_b64
                """,
                (
                    study["study_id"],
                    study["patient_id"],
                    study["modality"],
                    study["body_part"],
                    study.get("source", "UPLOAD"),
                    study.get("status", "PENDING_REVIEW"),
                    study.get("top_finding"),
                    study.get("confidence"),
                    json.dumps(study.get("predictions", [])),
                    json.dumps(study.get("report", {})),
                    study.get("raw_scan_b64") or study.get("scan_b64"),
                    study.get("scan_b64"),
                    study.get("heatmap_b64"),
                    study.get("overlay_b64"),
                    study.get("created_at", now),
                ),
            )

    def _hydrate(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        item["predictions"] = json.loads(item.pop("predictions_json") or "[]")
        item["report"] = json.loads(item.pop("report_json") or "{}")
        return item

    def list_studies(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies ORDER BY created_at DESC"
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def get_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()
        return self._hydrate(row) if row else None

    def record_decision(
        self,
        study_id: str,
        radiologist_id: str,
        decision: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        study = self.get_study(study_id)
        if not study:
            raise KeyError(f"Study {study_id} not found")
        decision_upper = decision.upper()
        if decision_upper not in ("APPROVED", "REJECTED"):
            raise ValueError("decision must be APPROVED or REJECTED")

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE studies SET status = ? WHERE study_id = ?",
                (decision_upper, study_id),
            )
            cur = conn.execute(
                """
                INSERT INTO audit_logs (
                    study_id, radiologist_id, decision, ai_finding,
                    ai_confidence, notes, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id,
                    radiologist_id,
                    decision_upper,
                    study.get("top_finding") or "Unknown",
                    float(study.get("confidence") or 0.0),
                    notes,
                    now,
                ),
            )
            log_id = cur.lastrowid

        return {
            "id": log_id,
            "study_id": study_id,
            "radiologist_id": radiologist_id,
            "decision": decision_upper,
            "ai_finding": study.get("top_finding") or "Unknown",
            "ai_confidence": float(study.get("confidence") or 0.0),
            "notes": notes,
            "decided_at": now,
        }

    def list_audit_logs(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

"""Milestone 5: Orthanc DICOM server polling worker that pulls new studies and triggers AI inference."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
import requests


class OrthancPoller:
    """Polls Orthanc `/changes` for StableStudy events and forwards instances to the FastAPI backend."""

    def __init__(
        self,
        orthanc_url: str = "http://localhost:8042",
        auth: tuple[str, str] = ("orthanc", "orthanc"),
        backend_url: str = "http://localhost:8000",
    ) -> None:
        self.orthanc_url = orthanc_url.rstrip("/")
        self.auth = auth
        self.backend_url = backend_url.rstrip("/")
        self.last_seq = 0

    def fetch_new_stable_studies(self, session: Optional[Any] = None) -> List[str]:
        """Query Orthanc /changes since last_seq and return newly stabilized Study IDs."""
        client = session or requests
        resp = client.get(
            f"{self.orthanc_url}/changes",
            params={"since": self.last_seq, "limit": 100},
            auth=self.auth,
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        self.last_seq = int(payload.get("Last", self.last_seq))

        studies = []
        for change in payload.get("Changes", []):
            if change.get("ChangeType") == "StableStudy":
                studies.append(change["ID"])
        return studies

    def process_study(self, orthanc_study_id: str, session: Optional[Any] = None) -> Dict[str, Any]:
        """Download preview PNG + DICOM tags from Orthanc and submit to FastAPI inference."""
        client = session or requests
        meta_resp = client.get(
            f"{self.orthanc_url}/studies/{orthanc_study_id}",
            auth=self.auth,
            timeout=10,
        )
        meta_resp.raise_for_status()
        study_info = meta_resp.json()

        instances_resp = client.get(
            f"{self.orthanc_url}/studies/{orthanc_study_id}/instances",
            auth=self.auth,
            timeout=10,
        )
        instances_resp.raise_for_status()
        instances = instances_resp.json()
        if not instances:
            raise ValueError(f"No DICOM instances found for study {orthanc_study_id}")

        first_instance_id = instances[0]["ID"]
        preview_resp = client.get(
            f"{self.orthanc_url}/instances/{first_instance_id}/preview",
            auth=self.auth,
            timeout=15,
        )
        preview_resp.raise_for_status()

        tags = study_info.get("MainDicomTags", {})
        patient_tags = study_info.get("PatientMainDicomTags", {})
        files = {"file": (f"{orthanc_study_id}.png", preview_resp.content, "image/png")}
        data = {
            "study_id": tags.get("StudyInstanceUID", orthanc_study_id),
            "patient_id": patient_tags.get("PatientID", "ORTHANC-ANON"),
            "modality": tags.get("Modality", "CR"),
            "body_part": tags.get("StudyDescription", "CHEST"),
            "source": "ORTHANC_PACS",
        }
        infer_resp = client.post(
            f"{self.backend_url}/studies/upload",
            data=data,
            files=files,
            timeout=30,
        )
        infer_resp.raise_for_status()
        return infer_resp.json()

    def run_forever(self, poll_interval_sec: int = 5) -> None:
        print(f"Starting Orthanc poller -> {self.orthanc_url} (interval={poll_interval_sec}s)")
        while True:
            try:
                for study_id in self.fetch_new_stable_studies():
                    print(f"[OrthancPoller] New StableStudy detected: {study_id}")
                    result = self.process_study(study_id)
                    print(
                        f"[OrthancPoller] Inferred {study_id}: "
                        f"{result.get('top_finding')} ({result.get('confidence')})"
                    )
            except Exception as exc:
                print(f"[OrthancPoller] Poll cycle warning: {exc}")
            time.sleep(poll_interval_sec)


if __name__ == "__main__":
    poller = OrthancPoller(
        orthanc_url=os.getenv("ORTHANC_URL", "http://localhost:8042"),
        auth=(os.getenv("ORTHANC_USER", "orthanc"), os.getenv("ORTHANC_PASS", "orthanc")),
        backend_url=os.getenv("BACKEND_URL", "http://localhost:8000"),
    )
    poller.run_forever(int(os.getenv("POLL_INTERVAL_SEC", "5")))

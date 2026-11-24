# REST API Documentation

Base URL (local): `http://localhost:8000` (interactive OpenAPI docs at `/docs`).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns service health, active inference engine mode, and educational disclaimer. |
| `GET` | `/studies` | Lists all ingested studies with AI findings, confidence scores, and review status. |
| `GET` | `/studies/{study_id}` | Retrieves a single study including base64 scan, Grad-CAM heatmap, and blended overlay. |
| `POST` | `/studies/upload` | Accepts multipart image upload (`file`, `study_id`, `patient_id`, `modality`, `body_part`), runs inference, and stores results. |
| `POST` | `/studies/{study_id}/review` | Records a radiologist `APPROVED` or `REJECTED` decision with `radiologist_id` and `notes` in the SQLite audit log. |
| `GET` | `/audit-log` | Returns the chronological radiologist decision audit trail. |

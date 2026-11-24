# AIML-02: AI-Assisted Medical Imaging and Diagnosis System

> **FOR EDUCATION ONLY — NOT FOR CLINICAL USE.**
> This software is an educational capstone prototype developed for the AI / Machine Learning specialization track (AIML-02). It has **not** been cleared or approved by any regulatory body (FDA, CE, CDSCO) and must **never** be used for real patient diagnosis or clinical decision-making.

An end-to-end medical imaging platform that ingests DICOM studies via an [Orthanc DICOM Server](https://www.orthanc-server.com/), runs deep learning classification (DenseNet-121 / EfficientNet) and 3D segmentation ([MONAI](https://project-monai.github.io/) 3D U-Net), generates [Grad-CAM](https://arxiv.org/abs/1610.02391) explanation heatmaps, and provides a React radiologist review interface with Approve/Reject controls and an immutable SQLite audit log.

## Architecture Overview

```
[DICOM Upload / Modality]
        │
        ▼
[Orthanc DICOM Server (:8042)] ──(REST /changes)──► [Orthanc Poller Worker]
                                                            │
                                                            ▼
                                            [FastAPI AI Inference Service (:8000)]
                                             ├── 2D Classifier (DenseNet-121) + Grad-CAM
                                             ├── 3D Segmentation (MONAI 3D U-Net)
                                             └── SQLite Audit Log & Study Store
                                                            │
                                                            ▼
                                            [React Radiologist Viewer (:5173 / :80)]
                                             ├── Scan + Heatmap / Mask Overlay Controls
                                             ├── Confidence Scores & Findings
                                             └── Approve / Reject + Audit Trail
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/API_DOCS.md](docs/API_DOCS.md), and [docs/MODEL_CARD_LIMITS_BIAS.md](docs/MODEL_CARD_LIMITS_BIAS.md) for full details.

## Supported Public Datasets

- **2D Classification:** [CheXpert (Stanford ML Group)](https://stanfordmlgroup.github.io/competitions/chexpert/) (224,316 chest radiographs, 65,240 patients) or [NIH ChestX-ray14 (NIH Clinical Center)](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) (112,120 frontal radiographs, 14 pathologies).
- **3D Segmentation:** [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/) (e.g., `Task06_Lung` CT or `Task01_BrainTumour` MRI via [AWS Open Data](https://registry.opendata.aws/msd/)).
- **Data Versioning:** Managed via [DVC](https://dvc.org/) (`dvc.yaml` + `params.yaml`).

## Quick Start (Docker Compose)

```bash
# 1. Clone the repository and start all services
docker compose up --build

# Services:
# - React Radiologist Viewer: http://localhost:3000
# - FastAPI Inference & Audit API: http://localhost:8000/docs
# - Orthanc DICOM Explorer: http://localhost:8042 (user: orthanc / pass: orthanc)
```

## Local Development & ML Pipeline

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 2. Run DVC pipeline (or run individual ML scripts)
dvc init
dvc repro

# 3. Start backend locally (runs in demo inference mode if model weights are not yet trained)
uvicorn backend.main:app --reload --port 8000

# 4. Start React frontend locally
cd frontend && npm install && npm run dev
```

## Running Automated Tests

```bash
python3 -m unittest discover -s tests -v
```

## Milestones Covered

1. Public dataset loaders (CheXpert / NIH ChestX-ray14 + MSD) & DICOM parsing (`ml/dicom_utils.py`).
2. Preprocessing (resize, CT/X-ray windowing, normalization, augmentation) & DVC versioning (`dvc.yaml`, `params.yaml`, `ml/preprocess.py`).
3. Transfer-learning 2D classifier (`ml/train_classifier.py`) & MONAI 3D U-Net (`ml/train_segmentation.py`).
4. Grad-CAM visual explanations (`ml/gradcam.py`).
5. Orthanc DICOM server configuration (`docker/orthanc.json`) & study polling worker (`backend/orthanc_poller.py`).
6. React radiologist review workstation with overlays, confidence scores, Approve/Reject workflow, and audit log (`frontend/src/App.jsx`, `backend/database.py`).
7. Docker containerization (`docker-compose.yml`, `docker/`) & model limits/bias documentation (`docs/MODEL_CARD_LIMITS_BIAS.md`).

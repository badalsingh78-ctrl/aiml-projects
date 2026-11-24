#!/usr/bin/env bash
set -euo pipefail

git init
git add .gitignore .dvcignore README.md docs/
git commit -m "docs: initialize AIML-02 project structure, README, and safety docs"

git add params.yaml dvc.yaml ml/
git commit -m "feat(ml): add DICOM utils, preprocessing, DenseNet-121, MONAI 3D U-Net, and Grad-CAM"

git add backend/ tests/
git commit -m "feat(backend): add FastAPI inference API, SQLite audit log, Orthanc poller, and tests"

git add frontend/ docker/ docker-compose.yml
git commit -m "feat(frontend,docker): add React radiologist viewer and Docker Compose stack"

echo "Git repository initialized with 4 clean milestone commits."

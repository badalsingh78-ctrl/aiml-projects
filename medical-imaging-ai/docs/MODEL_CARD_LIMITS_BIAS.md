# Model Card, Limitations & Bias Documentation

> **FOR EDUCATION ONLY — NOT FOR CLINICAL USE.**
> This system is an academic training prototype for AIML-02. It is **not** a medical device and must **not** be used for screening, diagnosis, triage, or treatment decisions.

## 1. Intended Use & Scope
- **Intended Use:** Educational demonstration of end-to-end medical imaging ML engineering (DICOM ingestion via [Orthanc](https://www.orthanc-server.com/), DVC pipelines, PyTorch/MONAI training, Grad-CAM visualization, and radiologist-in-the-loop audit workflows).
- **Out-of-Scope:** Clinical deployment, autonomous reporting, pediatric imaging, or modalities outside the training distribution.

## 2. Known Dataset & Model Biases
- **Institutional & Demographic Shift:** [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) (Stanford Health Care) and [NIH ChestX-ray14](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) reflect single-center U.S. hospital demographics and scanner protocols; performance degrades on external hospitals, portable X-ray units, and underrepresented age/sex/ethnicity cohorts.
- **NLP Label Noise:** Both CheXpert and NIH ChestX-ray14 derive labels from radiology reports via automated NLP labelers, introducing label noise and uncertainty markers.
- **Spurious Shortcuts (Clever Hans Effect):** Chest X-ray CNNs can latch onto chest tubes, ECG leads, portable-view markers, or hospital-specific text overlays rather than true pathology.
- **Class Imbalance:** Rare pathologies (e.g., hernia, pneumothorax, small lung nodules in [Medical Segmentation Decathlon](http://medicaldecathlon.com/)) suffer higher false-negative rates without calibrated thresholds.

## 3. Explainability & Safety Limits
- **Grad-CAM Resolution:** [Grad-CAM](https://arxiv.org/abs/1610.02391) maps coarse final convolutional feature maps (e.g., `7x7` upsampled to `224x224`), so heatmaps show approximate regions of influence rather than pixel-accurate lesion boundaries.
- **Automation Bias:** High-confidence AI scores can anchor human reviewers; therefore, the UI mandates explicit radiologist Approve/Reject verification and logs every decision with reviewer ID and timestamp.

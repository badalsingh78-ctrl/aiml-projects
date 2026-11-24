# One-Click Windows PowerShell Runner for CinePulse AI (AIML-03)
Write-Host "=== CinePulse AI: Large-Scale Recommendation Engine (AIML-03) ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
Write-Host "[2/4] Installing lightweight runtime dependencies..." -ForegroundColor Yellow
pip install -q -r requirements-lite.txt

$env:PYTHONPATH = "."
Write-Host "[3/4] Running temporal split, feature store & two-stage training..." -ForegroundColor Yellow
python data/prepare_movielens.py --synthetic --num-users 400
python spark/feature_engineering.py --force-local
python models/local_engine.py

Write-Host "[4/4] Launching CinePulse AI UI & FastAPI Backend at http://127.0.0.1:8000 ..." -ForegroundColor Green
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

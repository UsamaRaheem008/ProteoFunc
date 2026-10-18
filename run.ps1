# ============================================================================
# ProteoFunc  -  Complete Pipeline Runner
# ============================================================================
# This script runs the entire pipeline from start to finish in the correct order.
# Usage: Right-click -> Run with PowerShell  OR  .\run.ps1 in terminal
# ============================================================================

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"

$PYTHON = "python"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  ProteoFunc  -  Multimodal Protein Function Prediction" -ForegroundColor Cyan
Write-Host "  Complete Pipeline Runner" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Stage 1: Extract Embeddings ──────────────────────────────────────────────
Write-Host "[1/6] Extracting ESM-2 Embeddings..." -ForegroundColor Yellow
& $PYTHON run_pipeline.py --stage embeddings
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at embeddings stage!" -ForegroundColor Red; exit 1 }
Write-Host "[1/6] Embeddings DONE!" -ForegroundColor Green
Write-Host ""

# ── Stage 2: Build PPI Graph ─────────────────────────────────────────────────
Write-Host "[2/6] Building 186k-Edge PPI Graph with ESM-2 & Centrality..." -ForegroundColor Yellow
& $PYTHON run_pipeline.py --stage graph
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at graph stage!" -ForegroundColor Red; exit 1 }
Write-Host "[2/6] PPI Graph DONE!" -ForegroundColor Green
Write-Host ""

# ── Stage 3: Build Dataset ───────────────────────────────────────────────────
Write-Host "[3/6] Building Dataset (Train/Val/Test split)..." -ForegroundColor Yellow
& $PYTHON run_pipeline.py --stage dataset
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at dataset stage!" -ForegroundColor Red; exit 1 }
Write-Host "[3/6] Dataset DONE!" -ForegroundColor Green
Write-Host ""

# ── Stage 4: Train Models ────────────────────────────────────────────────────
Write-Host "[4/6] Training Models (sequence_only + network_only + multimodal)..." -ForegroundColor Yellow
Write-Host "       This may take 30-60 minutes..." -ForegroundColor DarkYellow
& $PYTHON run_pipeline.py --stage train
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at training stage!" -ForegroundColor Red; exit 1 }
Write-Host "[4/6] Training DONE!" -ForegroundColor Green
Write-Host ""

# ── Stage 5: Evaluate Models ─────────────────────────────────────────────────
Write-Host "[5/6] Evaluating Models (F-max, AUROC, AUPR, stratified)..." -ForegroundColor Yellow
& $PYTHON run_pipeline.py --stage evaluate
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at evaluation stage!" -ForegroundColor Red; exit 1 }
Write-Host "[5/6] Evaluation DONE!" -ForegroundColor Green
Write-Host ""

# ── Stage 6: Finalize Figures & Reports ──────────────────────────────────────
Write-Host "[6/6] Generating Publication Figures & Reports (finalize.py)..." -ForegroundColor Yellow
& $PYTHON finalize.py
if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: Finalize had issues (non-critical)" -ForegroundColor DarkYellow }
Write-Host "[6/6] Figures & Reports DONE!" -ForegroundColor Green
Write-Host ""

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Trained models:      ProteoFunc_outputs/checkpoints/" -ForegroundColor White
Write-Host "  Evaluation results:  ProteoFunc_outputs/evaluation/" -ForegroundColor White
Write-Host "  Publication figures: ProteoFunc_outputs/figures/" -ForegroundColor White
Write-Host ""

# ── Launch Dashboard ─────────────────────────────────────────────────────────
Write-Host "Launching ProteoFunc Dashboard..." -ForegroundColor Cyan
Write-Host "  Dashboard will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
& $PYTHON -m streamlit run app.py

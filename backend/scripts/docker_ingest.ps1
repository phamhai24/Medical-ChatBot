# Re-ingest straight into the Docker named volume (medical_rag_chatbot_data).
#
# Since docker-compose.yml mounts that volume at /app/data, running the ingest
# scripts INSIDE the running `api` container writes results directly into it.
# Running ingest natively on the host instead (python -m src.rag.ingest ...)
# would write to backend/data/ on Windows, which nothing here reads anymore —
# you'd have to re-copy it into the volume by hand. Always use this script (or
# `make docker-ingest`) for re-ingesting the Docker deployment.
#
# Usage:
#   .\scripts\docker_ingest.ps1            # full rebuild (chunk + embed + BM25)
#   .\scripts\docker_ingest.ps1 -NoRebuild  # incremental ingest, no BM25 rebuild

param(
    [switch]$NoRebuild
)

$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\..\.."
try {
    $running = docker compose ps --status running --format "{{.Service}}" 2>$null
    if ($running -notcontains "api") {
        throw "The 'api' container isn't running. Start it first: docker compose up -d"
    }

    $ingestArgs = @("--config", "config/rag_config.yaml")
    if (-not $NoRebuild) { $ingestArgs += "--rebuild" }

    Write-Host "==> Ingesting into the volume-backed vector store (this is the slow step)..." -ForegroundColor Cyan
    docker compose exec api python scripts/ingest_data.py @ingestArgs
    if ($LASTEXITCODE -ne 0) { throw "ingest_data.py failed (exit $LASTEXITCODE)" }

    if (-not $NoRebuild) {
        Write-Host "==> Rebuilding the whole-corpus BM25 index..." -ForegroundColor Cyan
        docker compose exec api python scripts/build_bm25_index.py
        if ($LASTEXITCODE -ne 0) { throw "build_bm25_index.py failed (exit $LASTEXITCODE)" }
    }

    Write-Host "==> Restarting api so the next warm-up picks up the new data..." -ForegroundColor Cyan
    docker compose restart api

    Write-Host "Done. Check: curl http://localhost:8000/health" -ForegroundColor Green
}
finally {
    Pop-Location
}

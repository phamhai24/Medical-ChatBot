"""Ingest data into the vector store (CLI script)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.pipeline import RAGPipeline
from src.utils.config_loader import load_config


def _project_processes() -> list[str]:
    """Best-effort Windows process check for apps that can hold Chroma open."""
    if os.name != "nt":
        return []

    try:
        import subprocess

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process | "
                    "Where-Object { $_.Name -match 'python|conda|streamlit|uvicorn|jupyter' } | "
                    "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []

    output = completed.stdout.strip()
    if not output:
        return []

    try:
        import json

        data = json.loads(output)
        rows = data if isinstance(data, list) else [data]
        current_pid = os.getpid()
        parent_pid = os.getppid()
        rows_by_pid = {int(row.get("ProcessId", 0)): row for row in rows}
        ancestor_pids = {current_pid, parent_pid}
        cursor = parent_pid
        while cursor in rows_by_pid:
            next_pid = int(rows_by_pid[cursor].get("ParentProcessId", 0))
            if next_pid in ancestor_pids or next_pid == 0:
                break
            ancestor_pids.add(next_pid)
            cursor = next_pid

        processes = []
        for row in rows:
            pid = int(row.get("ProcessId", 0))
            command = str(row.get("CommandLine", ""))
            if pid in ancestor_pids:
                continue
            if not any(
                marker in command
                for marker in (
                    "Medical_RAG_Chatbot",
                    "ingest_data",
                    "src\\api",
                    "src/api",
                    "src\\web",
                    "src/web",
                    "streamlit",
                    "jupyter",
                )
            ):
                continue
            processes.append(f"{pid} {row.get('Name')}: {command}")
        return processes
    except Exception:
        return [output]


def _preflight_rebuild(vector_config: dict) -> None:
    persist_dir = Path(vector_config.get("persist_directory", "data/vectorstore")).resolve()
    lock_path = persist_dir.parent / f".{persist_dir.name}.ingest.lock"
    staging_dirs = sorted(persist_dir.parent.glob(f"{persist_dir.name}.staging-*"))

    if lock_path.exists():
        raise RuntimeError(
            f"Found stale/active ingest lock: {lock_path}. "
            "Verify no ingest process is running before removing it."
        )

    if staging_dirs:
        names = ", ".join(str(path) for path in staging_dirs[:5])
        raise RuntimeError(
            "Found leftover staging vectorstore directories. "
            f"Remove them after verifying no ingest process is running: {names}"
        )

    processes = _project_processes()
    if processes:
        joined = "\n  ".join(processes)
        raise RuntimeError(
            "Other project Python/API/Streamlit/Jupyter processes are running and "
            "may hold ChromaDB files open. Stop them before rebuild:\n  " + joined
        )


def main():
    parser = argparse.ArgumentParser(description="Ingest medical Q&A data into vector store")
    parser.add_argument("--config", default="config/rag_config.yaml")
    parser.add_argument("--data", default="data/processed/data.json")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild index")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)
    vector_config = config.get("rag", {}).get("vector_store", {})
    embedding_config = config.get("rag", {}).get("embedding", {})

    if args.rebuild and vector_config.get("type") == "chroma":
        _preflight_rebuild(vector_config)

    try:
        import chromadb
        chroma_version = chromadb.__version__
    except Exception:
        chroma_version = "not installed"

    print(f"Ingesting data from: {args.data}")
    print(f"Config: {args.config}")
    print(f"Python: {sys.executable}")
    print(f"ChromaDB: {chroma_version}")
    print(f"Rebuild: {args.rebuild}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding model: {embedding_config.get('model_name')}")
    print(f"Embedding device: {embedding_config.get('device')}")
    print(f"Vector store: {vector_config.get('type')}")
    print(f"Persist directory: {vector_config.get('persist_directory')}")
    if args.rebuild and vector_config.get("type") == "chroma":
        print(
            "Safe rebuild: enabled (builds in staging, validates, then promotes atomically)"
        )
    print()

    stats = pipeline.ingest(
        data_path=args.data,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
        show_progress=not args.no_progress,
    )

    print()
    print("=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

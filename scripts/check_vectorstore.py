"""Diagnose a persisted Chroma vector store without rebuilding it."""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.vector_store import ChromaStore
from src.utils.config_loader import load_config


def inspect_sqlite(persist_dir: Path) -> dict:
    db_path = persist_dir / "chroma.sqlite3"
    result = {
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "journal_exists": (persist_dir / "chroma.sqlite3-journal").exists(),
        "segments": [],
    }
    if not db_path.exists():
        return result

    with sqlite3.connect(str(db_path), timeout=5.0) as conn:
        result["embeddings"] = conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        result["queue"] = conn.execute(
            "SELECT COUNT(*) FROM embeddings_queue"
        ).fetchone()[0]
        result["max_seq_id"] = conn.execute("SELECT * FROM max_seq_id").fetchall()
        result["segments"] = conn.execute(
            "SELECT id, type, scope, collection FROM segments"
        ).fetchall()
    return result


def inspect_hnsw_files(persist_dir: Path, segments: list[tuple]) -> list[dict]:
    required = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin")
    reports = []
    for segment_id, _type, scope, _collection in segments:
        if scope != "VECTOR":
            continue

        segment_dir = persist_dir / segment_id
        files = {}
        for name in required:
            path = segment_dir / name
            files[name] = path.stat().st_size if path.exists() else None

        reports.append({
            "segment_id": segment_id,
            "segment_dir": str(segment_dir),
            "exists": segment_dir.exists(),
            "files": files,
        })
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Chroma vectorstore health")
    parser.add_argument("--config", default="config/rag_config.yaml")
    parser.add_argument("--path", default=None, help="Override Chroma persist directory")
    parser.add_argument("--probe", action="store_true", help="Run Chroma count validation")
    args = parser.parse_args()

    config = load_config(args.config)
    vector_config = config.get("rag", {}).get("vector_store", {})
    persist_dir = Path(args.path or vector_config.get("persist_directory")).resolve()
    collection = vector_config.get("collection_name", "medical_qa")

    print(f"Persist directory: {persist_dir}")
    sqlite_report = inspect_sqlite(persist_dir)
    for key, value in sqlite_report.items():
        if key != "segments":
            print(f"{key}: {value}")

    print("segments:")
    for row in sqlite_report.get("segments", []):
        print(f"  {row}")

    print("hnsw_files:")
    for report in inspect_hnsw_files(persist_dir, sqlite_report.get("segments", [])):
        print(f"  segment: {report['segment_id']}")
        print(f"    exists: {report['exists']}")
        for name, size in report["files"].items():
            print(f"    {name}: {size}")

    if args.probe:
        print("chroma_probe:")
        if not sqlite_report.get("db_exists"):
            print("  status: missing")
            print("  error: chroma.sqlite3 does not exist; nothing to probe yet.")
            return 1

        store = ChromaStore(str(persist_dir), collection)
        try:
            store.load()
            print(f"  count: {store.count()}")
            print("  status: ok")
        except Exception as exc:
            print(f"  status: error")
            print(f"  error: {exc}")
            return 1
        finally:
            store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

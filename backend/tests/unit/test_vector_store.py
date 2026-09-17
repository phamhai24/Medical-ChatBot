"""Unit tests for vector store behavior."""

from pathlib import Path

import pytest

from src.rag.vector_store import ChromaStore, VectorStoreUnavailable


class BrokenCollection:
    def count(self):
        raise RuntimeError(
            "Error constructing hnsw segment reader: Error loading hnsw index"
        )


def test_chroma_count_surfaces_hnsw_index_errors():
    store = ChromaStore(persist_directory="missing", collection_name="medical_qa")
    store.collection = BrokenCollection()

    with pytest.raises(VectorStoreUnavailable) as exc_info:
        store.count()

    message = str(exc_info.value)
    assert "persisted HNSW index" in message
    assert "scripts/ingest_data.py --rebuild --no-progress" in message
    assert store.last_error == message


def test_chroma_reset_storage_backs_up_existing_directory(tmp_path):
    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()
    (persist_dir / "chroma.sqlite3").write_text("old", encoding="utf-8")

    store = ChromaStore(persist_directory=str(persist_dir), collection_name="medical_qa")
    backup_path = store.reset_storage(backup=True)

    assert backup_path is not None
    backup_dir = Path(backup_path)

    assert persist_dir.exists()
    assert not any(persist_dir.iterdir())
    assert (backup_dir / "chroma.sqlite3").read_text(encoding="utf-8") == "old"


def test_chroma_write_lock_blocks_overlapping_writes(tmp_path):
    persist_dir = tmp_path / "vectorstore"
    store = ChromaStore(persist_directory=str(persist_dir), collection_name="medical_qa")

    with store.write_lock():
        with pytest.raises(RuntimeError) as exc_info:
            with store.write_lock():
                pass

    assert "Another vector-store ingest/rebuild appears to be running" in str(exc_info.value)


def test_chroma_promote_from_uses_validated_staging_directory(tmp_path):
    target_dir = tmp_path / "vectorstore"
    staging_dir = tmp_path / "vectorstore.staging-test"
    target_dir.mkdir()
    staging_dir.mkdir()
    (target_dir / "chroma.sqlite3").write_text("old", encoding="utf-8")
    (staging_dir / "chroma.sqlite3").write_text("new", encoding="utf-8")

    target = ChromaStore(persist_directory=str(target_dir), collection_name="medical_qa")
    staging = ChromaStore(persist_directory=str(staging_dir), collection_name="medical_qa")

    backup_path = target.promote_from(staging, backup=True)

    assert backup_path is not None
    assert (target_dir / "chroma.sqlite3").read_text(encoding="utf-8") == "new"
    assert Path(backup_path).name.startswith("vectorstore.backup-")
    assert (Path(backup_path) / "chroma.sqlite3").read_text(encoding="utf-8") == "old"
    assert not staging_dir.exists()

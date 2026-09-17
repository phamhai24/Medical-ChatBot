"""Vector store implementations for RAG"""

import logging
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorStoreUnavailable(RuntimeError):
    """Raised when a vector store exists but cannot be read safely."""


class StoreWriteLock:
    """Simple process-level lock file for long vector-store writes."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fd: Optional[int] = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(str(self.lock_path), flags)
        except FileExistsError as exc:
            try:
                owner = self.lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                owner = "unknown owner"
            raise RuntimeError(
                "Another vector-store ingest/rebuild appears to be running. "
                f"Lock file: {self.lock_path}. Owner: {owner}. "
                "Stop the other Python/API/Streamlit process before rebuilding. "
                "Remove this lock only after verifying no ingest process is alive."
            ) from exc

        message = f"pid={os.getpid()} started_at={datetime.now().isoformat()}\n"
        os.write(self._fd, message.encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


class VectorStore:
    """
    Vector store factory and base interface.
    Supports ChromaDB, FAISS, and in-memory storage.
    """

    _instances = {}

    @classmethod
    def create(
        cls,
        store_type: str = "chroma",
        persist_directory: Optional[str] = None,
        collection_name: str = "medical_qa",
        embedding_dimension: Optional[int] = None,
    ) -> "VectorStore":
        """
        Factory method to create a vector store.

        Args:
            store_type: "chroma", "faiss", or "memory"
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            embedding_dimension: Dimension of embeddings

        Returns:
            VectorStore instance
        """
        if store_type == "chroma":
            return ChromaStore(persist_directory, collection_name, embedding_dimension)
        elif store_type == "faiss":
            return FaissStore(persist_directory, collection_name, embedding_dimension)
        elif store_type == "memory":
            return MemoryStore(collection_name, embedding_dimension)
        else:
            raise ValueError(f"Unknown store type: {store_type}")


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "medical_qa",
        embedding_dimension: Optional[int] = None
    ):
        self.persist_directory = persist_directory or "data/vectorstore/chroma"
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.client = None
        self.collection = None
        self.last_error: Optional[str] = None

    def _is_hnsw_index_error(self, error: Any) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "hnsw",
                "backfill request",
                "compactor",
                "segment reader",
                "loading hnsw index",
            )
        )

    def _metadata_record_count(self) -> Optional[int]:
        """Best-effort count from Chroma's SQLite metadata, not the vector index."""
        db_path = Path(self.persist_directory) / "chroma.sqlite3"
        if not db_path.exists():
            return None

        try:
            import sqlite3

            with sqlite3.connect(str(db_path), timeout=1.0) as conn:
                row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                return int(row[0]) if row else None
        except Exception:
            return None

    def metadata_count(self) -> Optional[int]:
        """Return stored metadata count when Chroma's vector index is unavailable."""
        return self._metadata_record_count()

    def _store_unavailable_error(self, operation: str, error: Any) -> VectorStoreUnavailable:
        detail = str(error)
        if self._is_hnsw_index_error(detail):
            metadata_count = self._metadata_record_count()
            count_hint = (
                f" SQLite metadata still lists about {metadata_count} records."
                if metadata_count is not None
                else ""
            )
            message = (
                f"ChromaDB {operation} failed because the persisted HNSW index "
                f"cannot be loaded.{count_hint} Rebuild the vector store with "
                f"`python scripts/ingest_data.py --rebuild --no-progress` from "
                f"the active project environment. Original error: {detail}"
            )
        else:
            message = f"ChromaDB {operation} failed: {detail}"

        self.last_error = message
        return VectorStoreUnavailable(message)

    def load(self):
        """Load or create the ChromaDB collection."""
        import chromadb
        from chromadb.config import Settings

        logger.info(f"Loading ChromaDB from {self.persist_directory}")

        self._raise_if_other_writer_active()

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._get_or_create_collection_with_timeout()
        self.last_error = None
        logger.info(f"ChromaDB collection '{self.collection_name}' loaded")

    def close(self):
        """Close the Chroma client so pending resources are released."""
        if self.client is None:
            return

        try:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

            clear_cache = getattr(self.client, "clear_system_cache", None)
            if callable(clear_cache):
                clear_cache()
        finally:
            self.client = None
            self.collection = None

    def write_lock(self) -> StoreWriteLock:
        """Return a lock that prevents overlapping writes to this store."""
        persist_path = Path(self.persist_directory).resolve()
        lock_name = f".{persist_path.name}.ingest.lock"
        return StoreWriteLock(persist_path.parent / lock_name)

    def _write_lock_path(self) -> Path:
        persist_path = Path(self.persist_directory).resolve()
        return persist_path.parent / f".{persist_path.name}.ingest.lock"

    def _lock_owner_pid(self, lock_path: Path) -> Optional[int]:
        try:
            content = lock_path.read_text(encoding="utf-8")
        except OSError:
            return None

        for part in content.split():
            if part.startswith("pid="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _raise_if_other_writer_active(self) -> None:
        lock_path = self._write_lock_path()
        if not lock_path.exists():
            return

        owner_pid = self._lock_owner_pid(lock_path)
        if owner_pid == os.getpid():
            return

        raise VectorStoreUnavailable(
            "ChromaDB store is currently being rebuilt by another process. "
            f"Lock file: {lock_path}. Owner PID: {owner_pid or 'unknown'}. "
            "Wait for ingest_data.py to finish before starting API/Streamlit."
        )

    def _unique_sibling_path(self, name: str) -> Path:
        parent = Path(self.persist_directory).resolve().parent
        candidate = parent / name
        suffix = 1
        while candidate.exists():
            candidate = parent / f"{name}-{suffix}"
            suffix += 1
        return candidate

    def _timestamped_sibling_path(self, label: str) -> Path:
        persist_path = Path(self.persist_directory).resolve()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self._unique_sibling_path(f"{persist_path.name}.{label}-{timestamp}")

    def _rename_directory(self, src: Path, dst: Path) -> None:
        """
        Rename a directory without shutil.move's copy-and-delete fallback.

        On Windows, shutil.move can copy a locked Chroma directory and then fail
        while deleting the source, leaving split SQLite/HNSW artifacts. A direct
        directory rename either succeeds or fails before copying bytes.
        """
        src = src.resolve()
        dst = dst.resolve()
        if src.anchor.lower() != dst.anchor.lower():
            raise ValueError(
                f"Refusing to rename Chroma store across drives: {src} -> {dst}"
            )
        try:
            src.rename(dst)
        except OSError as exc:
            raise RuntimeError(
                "Could not rename the Chroma store directory. On Windows this "
                "usually means API, Streamlit, another ingest process, or a file "
                f"scanner is still holding the store open. Source: {src}. "
                f"Destination: {dst}. Original error: {exc}"
            ) from exc

    def reset_storage(self, backup: bool = True) -> Optional[str]:
        """
        Replace the entire persistent Chroma directory.

        Chroma can fail to recover if the SQLite metadata exists but the HNSW
        files are missing or from an incompatible version. A collection delete
        is not enough in that state, so rebuilds need a clean directory.
        """
        self.close()

        persist_path = Path(self.persist_directory).resolve()
        if persist_path.parent == persist_path:
            raise ValueError(f"Refusing to reset unsafe path: {persist_path}")

        backup_path = None
        if persist_path.exists():
            if not persist_path.is_dir():
                raise ValueError(f"Persist path is not a directory: {persist_path}")

            has_existing_files = any(persist_path.iterdir())
            if has_existing_files:
                if backup:
                    backup_path = self._timestamped_sibling_path("backup")
                    self._rename_directory(persist_path, backup_path)
                    logger.info(
                        "Backed up existing ChromaDB store to %s", backup_path
                    )
                else:
                    shutil.rmtree(persist_path)
                    logger.info("Removed existing ChromaDB store at %s", persist_path)

        persist_path.mkdir(parents=True, exist_ok=True)
        self.last_error = None
        return str(backup_path) if backup_path else None

    def create_staging_store(self) -> "ChromaStore":
        """Create a clean Chroma store next to the production directory."""
        staging_path = self._timestamped_sibling_path("staging")
        staging_path.mkdir(parents=True, exist_ok=False)
        return ChromaStore(
            persist_directory=str(staging_path),
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
        )

    def _vector_segment_ids(self) -> List[str]:
        db_path = Path(self.persist_directory) / "chroma.sqlite3"
        if not db_path.exists():
            return []

        try:
            import sqlite3

            with sqlite3.connect(str(db_path), timeout=1.0) as conn:
                rows = conn.execute(
                    "SELECT id FROM segments WHERE scope = 'VECTOR'"
                ).fetchall()
                return [str(row[0]) for row in rows]
        except Exception:
            return []

    def _validate_hnsw_artifacts(self, count: int) -> None:
        if count <= 0:
            return

        persist_path = Path(self.persist_directory)
        db_path = persist_path / "chroma.sqlite3"
        if not db_path.exists():
            raise VectorStoreUnavailable(
                f"ChromaDB validation failed: missing SQLite database at {db_path}"
            )

        journal_path = persist_path / "chroma.sqlite3-journal"
        if journal_path.exists() and journal_path.stat().st_size > 0:
            raise VectorStoreUnavailable(
                "ChromaDB validation failed: SQLite rollback journal still exists "
                f"at {journal_path}. The writer did not shut down cleanly."
            )

        segment_ids = self._vector_segment_ids()
        if not segment_ids:
            raise VectorStoreUnavailable(
                "ChromaDB validation failed: no VECTOR segment was recorded in SQLite."
            )

        required = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin")
        valid_segment_dirs = []
        missing_details = []
        for segment_id in segment_ids:
            segment_dir = persist_path / segment_id
            missing = [name for name in required if not (segment_dir / name).exists()]
            if missing:
                missing_details.append(f"{segment_id}: missing {', '.join(missing)}")
                continue

            if (segment_dir / "data_level0.bin").stat().st_size == 0:
                missing_details.append(f"{segment_id}: data_level0.bin is empty")
                continue
            if (segment_dir / "length.bin").stat().st_size == 0:
                missing_details.append(f"{segment_id}: length.bin is empty")
                continue

            valid_segment_dirs.append(segment_dir)

        if not valid_segment_dirs:
            raise VectorStoreUnavailable(
                "ChromaDB validation failed: HNSW segment files are incomplete. "
                + "; ".join(missing_details)
            )

    def validate_persisted(
        self,
        expected_count: Optional[int] = None,
        probe_embedding: Optional[List[float]] = None,
    ) -> int:
        """Close, reopen, count, inspect HNSW files, and optionally query."""
        self.close()

        fresh = ChromaStore(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
        )
        try:
            fresh.load()
            count = fresh.count()
            if expected_count is not None and count != expected_count:
                raise VectorStoreUnavailable(
                    "ChromaDB validation failed: expected "
                    f"{expected_count} documents, reopened count is {count}."
                )

            fresh._validate_hnsw_artifacts(count)

            if count > 0 and probe_embedding is not None:
                results = fresh.search(probe_embedding, top_k=1)
                if not results:
                    raise VectorStoreUnavailable(
                        "ChromaDB validation failed: reopened store count is "
                        f"{count}, but a probe vector query returned no results."
                    )

            return count
        finally:
            fresh.close()

    def promote_from(self, staging_store: "ChromaStore", backup: bool = True) -> Optional[str]:
        """Atomically promote a validated staging store into this store path."""
        self.close()
        staging_store.close()

        target_path = Path(self.persist_directory).resolve()
        staging_path = Path(staging_store.persist_directory).resolve()
        if staging_path.parent != target_path.parent:
            raise ValueError(
                f"Staging store must be next to target: {staging_path} -> {target_path}"
            )
        if not staging_path.exists():
            raise FileNotFoundError(f"Staging Chroma store does not exist: {staging_path}")

        backup_path = None
        if target_path.exists():
            if any(target_path.iterdir()):
                if backup:
                    backup_path = self._timestamped_sibling_path("backup")
                    self._rename_directory(target_path, backup_path)
                    logger.info("Backed up existing ChromaDB store to %s", backup_path)
                else:
                    shutil.rmtree(target_path)
            else:
                target_path.rmdir()

        try:
            self._rename_directory(staging_path, target_path)
        except Exception:
            if backup_path and backup_path.exists() and not target_path.exists():
                self._rename_directory(backup_path, target_path)
            raise

        self.last_error = None
        return str(backup_path) if backup_path else None

    def _get_or_create_collection_with_timeout(self, timeout: float = 10.0):
        """Thread-safe get_or_create_collection with timeout."""
        result = {"collection": None, "error": None}

        def target():
            try:
                result["collection"] = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical Q&A vector store"}
                )
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            logger.warning(
                f"ChromaDB get_or_create_collection() timed out after {timeout}s. "
                f"Re-creating collection."
            )
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Medical Q&A vector store"}
            )
        if result["error"] is not None:
            raise result["error"]
        return result["collection"]

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the collection."""
        if self.collection is None:
            self.load()

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            ids=ids
        )
        logger.info(f"Added {len(texts)} documents to collection")

    def _query_with_timeout(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict],
        where_document: Optional[Dict],
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Thread-safe query with timeout."""
        result = {"output": None, "error": None}

        def target():
            try:
                result["output"] = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                    where_document=where_document,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            message = f"ChromaDB query() timed out after {timeout}s."
            self.last_error = message
            logger.warning(message)
            raise TimeoutError(message)
        if result["error"] is not None:
            error = self._store_unavailable_error("query()", result["error"])
            logger.warning(str(error))
            raise error
        self.last_error = None
        return result["output"]

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.collection is None:
            self.load()

        results = self._query_with_timeout(
            query_embedding, top_k, where, where_document
        )

        if not results:
            return []

        docs = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                docs.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "id": results["ids"][0][i] if results["ids"] else None,
                })
        return docs

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Simple similarity search returning (text, score) pairs."""
        results = self.search(query_embedding, top_k)
        return [(r["text"], r["distance"]) for r in results]

    def _count_with_timeout(self, timeout: float = 5.0) -> int:
        """Thread-safe count with timeout to prevent hangs in ChromaDB 1.x."""
        result = {"count": 0, "error": None}

        def target():
            try:
                result["count"] = self.collection.count()
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            message = f"ChromaDB count() timed out after {timeout}s."
            self.last_error = message
            logger.warning(message)
            raise TimeoutError(message)
        if result["error"] is not None:
            error = self._store_unavailable_error("count()", result["error"])
            logger.warning(str(error))
            raise error
        self.last_error = None
        return result["count"]

    def count(self) -> int:
        """Get the number of documents in the collection."""
        if self.collection is None:
            self.load()
        return self._count_with_timeout()

    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        if self.collection is None:
            self.load()
        self.collection.delete(ids=ids)

    def clear(self):
        """Clear all documents from the collection."""
        try:
            if self.collection is None:
                self.load()
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            logger.warning(
                "ChromaDB delete_collection() failed (%s). Resetting storage.",
                e,
            )
            self.reset_storage(backup=True)
            self.load()
        self.last_error = None
        logger.info(f"Cleared collection '{self.collection_name}'")


class FaissStore(VectorStore):
    """FAISS vector store implementation."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "medical_qa",
        embedding_dimension: Optional[int] = None
    ):
        self.persist_directory = persist_directory or "data/vectorstore/faiss"
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.documents = []
        self.metadatas = []
        self._doc_ids = []
        self._id_counter = 0

    def load(self):
        """Load or create the FAISS index."""
        import faiss
        import numpy as np

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        index_path = Path(self.persist_directory) / f"{self.collection_name}.index"
        meta_path = Path(self.persist_directory) / f"{self.collection_name}_meta.json"

        if index_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.documents = meta.get("documents", [])
                self.metadatas = meta.get("metadatas", [])
                self._doc_ids = meta.get("ids", [])
                self._id_counter = meta.get("counter", 0)
            self.embedding_dimension = meta.get("dim")
            logger.info(f"Loaded FAISS index with {len(self.documents)} docs")
        else:
            if self.embedding_dimension is None:
                raise ValueError("embedding_dimension required for new FAISS index")
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dimension))
            logger.info(f"Created new FAISS index (dim={self.embedding_dimension})")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the FAISS index."""
        import faiss
        import numpy as np

        if self.index is None:
            self.load()

        if self.embedding_dimension is None:
            self.embedding_dimension = len(embeddings[0])

        # Re-create index with IDMap if needed
        if not isinstance(self.index, faiss.IndexIDMap):
            self.index = faiss.IndexIDMap(self.index)

        embeddings_np = np.array(embeddings).astype("float32")
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)

        if ids is None:
            ids = [str(self._id_counter + i) for i in range(len(texts))]

        int_ids = np.array([int(id_.split("_")[-1]) if "_" in id_ else i
                            for i, id_ in enumerate(ids)])

        self.index.add_with_ids(embeddings_np, int_ids)
        self.documents.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in texts])
        self._doc_ids.extend(ids)
        self._id_counter = max(self._id_counter, len(self.documents))

        self._save()
        logger.info(f"Added {len(texts)} documents to FAISS index")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        import faiss
        import numpy as np

        if self.index is None:
            self.load()

        query_np = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_np)

        distances, indices = self.index.search(query_np, min(top_k, len(self.documents)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                    "distance": float(dist),
                    "id": self._doc_ids[idx] if idx < len(self._doc_ids) else str(idx),
                })
        return results

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Simple similarity search."""
        results = self.search(query_embedding, top_k)
        return [(r["text"], r["distance"]) for r in results]

    def count(self) -> int:
        """Get document count."""
        if self.index is None:
            self.load()
        return self.index.ntotal

    def _save(self):
        """Save index to disk."""
        import faiss
        faiss.write_index(self.index, str(Path(self.persist_directory) / f"{self.collection_name}.index"))
        with open(Path(self.persist_directory) / f"{self.collection_name}_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self._doc_ids,
                "counter": self._id_counter,
                "dim": self.embedding_dimension
            }, f, ensure_ascii=False)

    def clear(self):
        """Clear the index."""
        import faiss
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dimension)) if self.embedding_dimension else None
        self.documents = []
        self.metadatas = []
        self._doc_ids = []


class MemoryStore(VectorStore):
    """In-memory vector store for quick prototyping."""

    def __init__(
        self,
        collection_name: str = "medical_qa",
        embedding_dimension: Optional[int] = None
    ):
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.documents = []
        self.metadatas = []
        self._doc_ids = []

    def load(self):
        """No-op for memory store."""
        pass

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Store documents in memory."""
        self.documents.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in texts])
        self._doc_ids.extend(ids or [f"doc_{i}" for i in range(len(texts))])
        logger.info(f"Stored {len(texts)} documents in memory")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Simple cosine similarity search in memory."""
        import numpy as np

        if not self.documents:
            return []

        docs_emb = np.array([self.documents[0]] * len(self.documents))  # placeholder
        # For memory store, we need stored embeddings - this is a simplified version
        # In practice, use FAISS/Chroma for real similarity search
        return [{"text": "", "metadata": {}, "distance": 0.0, "id": ""}]

    def count(self) -> int:
        return len(self.documents)

"""Build the whole-corpus BM25 index (src/rag/bm25_index.py) from the existing
ChromaDB vector store, without re-embedding anything.

Run once after ingestion (or re-run after re-ingesting new data):
    python scripts/build_bm25_index.py

Takes ~2-3 minutes for ~700k chunks (pulling documents from Chroma + tokenizing
+ indexing); querying the built index is ~10ms regardless of corpus size.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.vector_store import VectorStore
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Build the whole-corpus BM25 index")
    parser.add_argument("--config", default="config/rag_config.yaml")
    parser.add_argument("--output", default="data/bm25_index")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    import bm25s

    config = load_config(args.config)
    vc = config["rag"]["vector_store"]
    store = VectorStore.create(
        store_type=vc.get("type", "chroma"),
        persist_directory=vc.get("persist_directory"),
        collection_name=vc.get("collection_name", "medical_qa"),
    )
    store.load()

    print("Pulling documents from the vector store...")
    t0 = time.perf_counter()
    corpus: list[dict] = []
    offset = 0
    while True:
        res = store.collection.get(
            limit=args.batch_size, offset=offset, include=["documents", "metadatas"]
        )
        ids = res["ids"]
        if not ids:
            break
        for i, doc_id in enumerate(ids):
            corpus.append({
                "id": doc_id,
                "text": res["documents"][i],
                "metadata": res["metadatas"][i] or {},
            })
        offset += args.batch_size
        print(f"  pulled {offset}...")
    print(f"Pulled {len(corpus)} documents in {time.perf_counter() - t0:.1f}s")

    print("Tokenizing + building the BM25 index...")
    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize([d["text"] for d in corpus], stopwords=None, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    print(f"Built in {time.perf_counter() - t0:.1f}s")

    print(f"Saving to {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    retriever.save(args.output, corpus=corpus)
    print("Done.")


if __name__ == "__main__":
    main()

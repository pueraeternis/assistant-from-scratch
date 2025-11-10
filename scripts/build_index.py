import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MD_SOURCE_PATH = Path("data/processed_markdown")
INDEX_OUTPUT_PATH = Path("data/vector_index")
INDEX_FILE = INDEX_OUTPUT_PATH / "faiss_index.bin"
CHUNKS_FILE = INDEX_OUTPUT_PATH / "chunks_with_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 1000  # Max chunk size in characters
CHUNK_OVERLAP = 150  # Overlap between chunks (usually 10-15% of chunk size)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def split_text_into_chunks(text: str, source_name: str) -> List[Dict[str, Any]]:
    """
    Splits a large text into fixed-size chunks with overlap.
    """
    chunks = []
    if not text:
        return []

    start_index = 0
    while start_index < len(text):
        end_index = start_index + CHUNK_SIZE
        chunk_text = text[start_index:end_index]

        chunks.append({"page_content": chunk_text, "metadata": {"source": source_name}})

        start_index += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def build_vector_index() -> None:
    """
    Builds a FAISS vector index from processed Markdown files.
    """
    INDEX_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logging.info("Index output directory '%s' is ready.", INDEX_OUTPUT_PATH)

    md_files = list(MD_SOURCE_PATH.glob("*.md"))
    if not md_files:
        logging.warning("No Markdown files found in '%s'.", MD_SOURCE_PATH)
        return

    logging.info("Found %d files for indexing.", len(md_files))

    all_chunks = []
    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        chunks = split_text_into_chunks(content, md_path.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        logging.error("Failed to create any chunks from the files.")
        return

    chunk_texts = [chunk["page_content"] for chunk in all_chunks]
    logging.info("Created %d chunks for vectorization.", len(chunk_texts))

    # --- Embedding creation ---
    logging.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    logging.info("Generating embeddings... This may take a while.")
    embeddings_list = model.encode(chunk_texts, show_progress_bar=True)
    embeddings = np.ascontiguousarray(np.array(embeddings_list, dtype=np.float32))
    vector_dimension = embeddings.shape[1]
    logging.info("Embeddings generated. Vector dimension: %d.", vector_dimension)

    # --- FAISS index creation ---
    logging.info("Creating and populating FAISS index...")
    index: Any = faiss.IndexFlatL2(vector_dimension)  # type: ignore
    index.add(embeddings)  # type: ignore
    logging.info("Added %d vectors to the index.", index.ntotal)

    # --- Save index ---
    faiss.write_index(index, str(INDEX_FILE))
    logging.info("FAISS index saved to '%s'.", INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    logging.info("Chunks with metadata saved to '%s'.", CHUNKS_FILE)

    logging.info("Done. Indexing completed successfully.")


if __name__ == "__main__":
    build_vector_index()

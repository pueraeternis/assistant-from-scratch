# tools/vector_search.py

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.base_tool import BaseTool
from core.logging import get_logger, setup_logging

# --- Configuration ---
INDEX_PATH = Path("data/vector_index")
INDEX_FILE = INDEX_PATH / "faiss_index.bin"
CHUNKS_FILE = INDEX_PATH / "chunks_with_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# configure logging
setup_logging()
logger = get_logger(__name__)


class VectorSearchTool(BaseTool):
    """
    Tool for semantic search over a local knowledge base
    using vector embeddings of scientific papers.
    """

    name = "KnowledgeBaseSearch"
    description = (
        "Use this tool to find information about Large Language Models (LLM), Agentic AI, "
        "and related topics from a specialized knowledge base of scientific papers. "
        "Provide a clear, specific question as input using the 'query' parameter."
    )

    def __init__(self):
        super().__init__()
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.model = None
        self._load_index()

    def _load_index(self) -> None:
        """Load resources (FAISS index, chunks, embedding model) on initialization."""
        try:
            logger.info("Loading Knowledge Base Search Tool resources...")
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Knowledge Base loaded. Index contains %d vectors.", self.index.ntotal)
        except FileNotFoundError as e:
            logger.error("Failed to load Knowledge Base: %s. Run scripts/build_index.py first.", e)
        except Exception as e:
            logger.error("Unexpected error loading Knowledge Base: %s", e)

    def _run(self, **kwargs: Any) -> str:
        """
        Perform semantic search over the vector database.
        Expects 'query' as a keyword argument.
        """
        if not self.index or not self.model:
            return "Error: Knowledge Base is not loaded. Check startup logs for errors."

        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return "Error: The 'query' argument is missing or is not a string for KnowledgeBaseSearch."

        logger.info("Performing knowledge base search for query: '%s'", query)

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = 3  # Number of top chunks to retrieve
        _, indices = self.index.search(query_embedding, k)

        if not indices.size > 0:
            return "No relevant information found in the knowledge base."

        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                chunk = self.chunks[i]
                source = chunk["metadata"].get("source", "Unknown source")
                content = chunk["page_content"]
                results.append(f"Source: {source}\nContent: {content}\n---")

        return "\n".join(results) if results else "No relevant content found for the given query."

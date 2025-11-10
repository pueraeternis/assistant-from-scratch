import re
from pathlib import Path

import arxiv

from core.logging import get_logger, setup_logging

# --- Configuration ---
# We are searching in categories: AI, Computation and Language, Machine Learning
# And using keywords: "Large Language Model", "LLM", "Agentic AI"
SEARCH_QUERY = 'cat:cs.AI OR cat:cs.CL OR cat:cs.LG AND ("Large Language Model" OR "LLM" OR "Agentic AI")'
MAX_RESULTS = 15
DOWNLOAD_PATH = Path("data/papers")

# configure logging
setup_logging()
logger = get_logger(__name__)


def sanitize_filename(title: str) -> str:
    """Sanitize a paper title to create a safe filename."""
    sanitized = title.replace(" ", "_")
    sanitized = re.sub(r"[^\w\-_.]", "", sanitized)
    return sanitized[:150]


def download_arxiv_papers():
    """
    Search recent arXiv papers by query and download their PDFs.
    """
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Directory '%s' is ready.", DOWNLOAD_PATH)

    # Create a search query
    # Sort by submission date to get the newest papers
    search = arxiv.Search(query=SEARCH_QUERY, max_results=MAX_RESULTS, sort_by=arxiv.SortCriterion.SubmittedDate)

    logger.info("Fetching top %s latest papers for query: '%s'...", MAX_RESULTS, SEARCH_QUERY)

    # Iterate over results and download PDFs
    papers_downloaded = 0
    client = arxiv.Client()
    for result in client.results(search):
        try:
            filename = sanitize_filename(result.title)

            # Skip if the file already exists
            if (DOWNLOAD_PATH / f"{filename}.pdf").exists():
                logger.info("Paper '%s' already exists. Skipping.", result.title)
                continue

            logger.info("Downloading: '%s'", result.title)
            result.download_pdf(dirpath=str(DOWNLOAD_PATH), filename=f"{filename}.pdf")
            papers_downloaded += 1

        except Exception as e:
            logger.error("Failed to download paper '%s': %s", result.title, e)

    logger.info("Done. Downloaded %s new papers.", papers_downloaded)


if __name__ == "__main__":
    download_arxiv_papers()

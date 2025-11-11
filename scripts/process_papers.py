# scripts/process_papers.py

import logging
import traceback
from pathlib import Path

import pymupdf4llm

PDF_SOURCE_PATH = Path("data/papers")
MD_OUTPUT_PATH = Path("data/processed_markdown")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_pdfs_to_markdown() -> None:
    """
    Iterates through all PDF files, extracts their text as Markdown
    using pymupdf4llm, and saves it into .md files.
    """
    MD_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logging.info("Markdown output directory '%s' is ready.", MD_OUTPUT_PATH)

    pdf_files = list(PDF_SOURCE_PATH.glob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDF files found in '%s'.", PDF_SOURCE_PATH)
        return

    logging.info("Found %d PDF files to process.", len(pdf_files))

    processed_count = 0
    for pdf_path in pdf_files:
        try:
            md_filename = pdf_path.stem + ".md"
            md_path = MD_OUTPUT_PATH / md_filename

            if md_path.exists():
                logging.info("File '%s' already exists. Skipping.", md_filename)
                continue

            logging.info("Processing '%s' into Markdown...", pdf_path.name)

            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
            md_path.write_bytes(markdown_text.encode("utf-8"))
            processed_count += 1

        except Exception:
            error_details = traceback.format_exc()
            logging.error("Failed to process file '%s'. Full traceback:\n%s", pdf_path.name, error_details)

    logging.info("Done. Processed %d new files into Markdown.", processed_count)


if __name__ == "__main__":
    process_pdfs_to_markdown()

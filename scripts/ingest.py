"""Ingest documents into the RAG pipeline.

Usage:
    python scripts/ingest.py --config configs/baseline.yaml --data-dir data/raw
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", required=True, help="Path to directory containing PDFs")
    args = parser.parse_args()

    from src.core.pipeline import RAGPipeline
    from src.ingestion import PDFParser, TextPreprocessor

    # Parse PDFs
    pdf_parser = PDFParser()
    docs = pdf_parser.parse_directory(args.data_dir)
    if not docs:
        logger.warning("No PDF documents found in %s", args.data_dir)
        sys.exit(0)

    # Preprocess text
    preprocessor = TextPreprocessor()
    for doc in docs:
        doc.content = preprocessor.preprocess(doc.content)

    # Ingest into pipeline
    pipeline = RAGPipeline.from_config(args.config)
    pipeline.ingest_documents(docs)

    logger.info("Ingested %d documents", len(docs))


if __name__ == "__main__":
    main()

"""PDF document parsing."""
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.core.document import Document

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF documents into text."""

    def parse(self, file_path: str | Path) -> Document:
        """Parse a PDF file into a Document.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Document with extracted text and metadata.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text_parts = []
        
        with fitz.open(file_path) as doc:
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "title": doc.metadata.get("title", file_path.stem),
                "author": doc.metadata.get("author", ""),
                "page_count": len(doc),
            }
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        
        return Document(content=full_text, metadata=metadata)
    
    def parse_directory(self, directory: str | Path) -> list[Document]:
        """Parse all PDF files in a directory.
        
        Args:
            directory: Path to directory containing PDFs.
            
        Returns:
            List of Documents.
        """
        directory = Path(directory)
        documents = []
        
        for pdf_path in directory.glob("*.pdf"):
            try:
                doc = self.parse(pdf_path)
                documents.append(doc)
            except Exception as e:
                logger.error("Error parsing %s: %s", pdf_path, e)
        
        return documents

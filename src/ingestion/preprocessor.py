"""Text preprocessing utilities."""
import re
from typing import Any


class TextPreprocessor:
    """Clean and preprocess text for RAG pipeline."""
    
    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        remove_page_numbers: bool = True,
        min_line_length: int = 10,
    ):
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.remove_page_numbers = remove_page_numbers
        self.min_line_length = min_line_length
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to text.
        
        Args:
            text: Raw text to preprocess.
            
        Returns:
            Cleaned text.
        """
        if self.remove_page_numbers:
            text = self._remove_page_numbers(text)
        
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)
        
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split("\n")
        lines = [line for line in lines if len(line.strip()) >= self.min_line_length or line.strip() == ""]
        text = "\n".join(lines)
        
        return text.strip()
    
    def _remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers."""
        # Remove lines that are just numbers (page numbers)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip if line is just a number or "Page X" pattern
            if stripped.isdigit():
                continue
            if re.match(r'^(página|page|pág\.?)\s*\d+$', stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns."""
        # Remove common footer patterns
        patterns = [
            r'^\s*\d+\s*$',  # Just page numbers
            r'^\s*-\s*\d+\s*-\s*$',  # - 1 -
            r'^\s*\[\s*\d+\s*\]\s*$',  # [1]
        ]
        
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            skip = False
            for pattern in patterns:
                if re.match(pattern, line):
                    skip = True
                    break
            if not skip:
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        return text

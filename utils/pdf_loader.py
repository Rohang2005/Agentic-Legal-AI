"""
PDF loader for legal judgment documents.

Uses PyPDF to extract raw text from PDF files.
"""

from pathlib import Path
from typing import Union

from pypdf import PdfReader


def load_pdf_text(path: Union[str, Path]) -> str:
    """
    Extract text from a PDF file.

    Args:
        path: File path to the PDF (string or Path).

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF has no extractable text.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.append(text)

    if not chunks:
        raise ValueError(f"No text could be extracted from PDF: {path}")

    return "\n\n".join(chunks)

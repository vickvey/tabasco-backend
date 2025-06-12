import re
from pathlib import Path

import pdfplumber
from pdfplumber.pdf import PDF
from pdfminer.pdfparser import PDFSyntaxError

ALLOWED_EXTENSIONS = {"pdf", "txt"}

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def standardize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove emails and mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove special characters, symbols, and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove single characters (like 'a', 'b', etc.)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def pdf2text(file_path: Path) -> str:
    """
    Extract text from a PDF file using pdfplumber, handling CropBox explicitly.

    Args:
        file_path (Path): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF, is invalid, or text extraction fails.
    """
    # Convert and validate file path
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    try:
        text = []
        with pdfplumber.open(file_path) as pdf:
            # Validate PDF structure
            if not isinstance(pdf, PDF) or not pdf.pages:
                raise ValueError("Invalid PDF: No pages found")

            for page in pdf.pages:
                # Explicitly handle CropBox: set to MediaBox if missing
                if page.cropbox is None:
                    if page.mediabox is None:
                        raise ValueError(f"Page {page.page_number}: Missing both CropBox and MediaBox")
                    page.cropbox = page.mediabox  # Set CropBox to MediaBox

                # Extract text
                page_text = page.extract_text()
                if page_text:
                    # Normalize whitespace
                    cleaned_text = " ".join(page_text.split())
                    text.append(cleaned_text)

        # Join pages' text or return empty string if no text
        return " ".join(text).strip() if text else ""

    except PDFSyntaxError as e:
        raise ValueError(f"Invalid PDF syntax: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
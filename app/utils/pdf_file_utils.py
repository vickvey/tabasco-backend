from pathlib import Path

import fitz # PyMuPDF
from fastapi import HTTPException

from app.config import settings

# Mount constants from settings
UPLOAD_FOLDER = settings.UPLOAD_FOLDER
ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf2text(file_path: Path) -> str:
        """
        Extract text from a PDF file using PyMuPDF (fitz).

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            str: Extracted text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If not a PDF or unreadable.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid file type (expected PDF): {file_path}")

        try:
            with fitz.open(file_path) as doc:
                text = " ".join(page.get_text("text") for page in doc if page.get_text("text"))
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

def read_file_text(file_path: Path) -> str:
    """
    Read text from a Path. Convert PDF via pdf2text or read as UTF-8 text.
    """
    try:
        return pdf2text(file_path) if file_path.suffix.lower() == ".pdf" else file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

def ensure_uploaded_file_exists(filename: str) -> Path:
    """
    Ensure the file exists in UPLOAD_FOLDER. Returns the full Path or raises HTTPException 404.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return file_path

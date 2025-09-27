from pathlib import Path
import fitz  # PyMuPDF

# Constants from settings
# ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS
# UPLOAD_FOLDER: Path = settings.UPLOAD_FOLDER

def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    """
    Check if the file has an allowed extension.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )

def pdf2text(file_path: Path) -> str:
    """
    Extract plain text from a PDF file using PyMuPDF.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file type (expected PDF): {file_path}")

    try:
        text_chunks: list[str] = []
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text("text")  # type: ignore
                if page_text:
                    text_chunks.append(page_text.strip())
        return "\n\n".join(text_chunks).strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

def read_file_text(file_path: Path) -> str:
    """
    Read the content of a file.
    - If PDF, extract text with PyMuPDF.
    - If TXT, read as UTF-8 text.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path.suffix.lower() == ".pdf":
            return pdf2text(file_path)
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise Exception(f"Failed to read file '{file_path.name}': {str(e)}")

def ensure_uploaded_file_exists(file_path: Path):
    """
    Ensure the file exists in the upload folder.
    Returns the full Path or raises FileNotFoundError.
    """
    # file_path = UPLOAD_FOLDER / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"'{file_path}' is not valid.")
    return file_path

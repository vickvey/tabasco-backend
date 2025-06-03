import re
from pathlib import Path

import pdfplumber

ALLOWED_EXTENSIONS = {"pdf", "txt"}

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def standardize_text(text):
    """
    Perform simple text cleaning.
    """
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def pdf2text(file_path: Path) -> str:
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += " ".join(page_text.split()) + " "
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
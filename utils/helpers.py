import re
from PyPDF2 import PdfReader

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    ALLOWED_EXTENSIONS = {"pdf", "txt"}
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

def pdf2text(file_path):
    """
    Convert a PDF file into a plain text string.
    """
    reader = PdfReader(file_path)
    text_list = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_list.append(page_text)
    return " ".join(text_list)
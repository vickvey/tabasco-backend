import re

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

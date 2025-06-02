# services/text_processing.py
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, FreqDist
from config.settings import UPLOAD_FOLDER
from utils.helpers import pdf2text

def get_noun_list(filename, top_n=50):
    file_path = UPLOAD_FOLDER / filename
    text = pdf2text(file_path) if filename.endswith(".pdf") else file_path.read_text(encoding="utf-8")
    # Implementation for extracting nouns
    pass

def process_sentences(text, target_word, frequency_limit=100):
    # Implementation for processing sentences
    pass
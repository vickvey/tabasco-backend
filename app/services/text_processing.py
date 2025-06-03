# services/text_processing.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, FreqDist
from app.settings import UPLOAD_FOLDER
from app.utils.helpers import pdf2text

# Ensure necessary NLTK packages are downloaded.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def get_noun_list(filename, top_n=50):
    file_path = UPLOAD_FOLDER / filename
    text = pdf2text(file_path) if filename.endswith(".pdf") else file_path.read_text(encoding="utf-8")
    tokens = word_tokenize(text)
    # Filter tokens to only alphabetic words.
    tokens = [t for t in tokens if t.isalpha()]
    # Get frequency distribution.
    freq_dist = FreqDist(tokens)
    # Tag tokens with POS.
    tagged_tokens = pos_tag(list(freq_dist.keys()))
    # Filter out nouns (POS starting with 'NN').
    nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]
    # Sort nouns by frequency (highest first).
    sorted_nouns = sorted(nouns, key=lambda x: freq_dist[x], reverse=True)
    return sorted_nouns[:top_n]

def process_sentences(text, target_word, frequency_limit=100):
    sentences = sent_tokenize(text)
    target_sentences = [sent for sent in sentences if target_word.lower() in sent.lower()]
    if len(target_sentences) > frequency_limit:
        target_sentences = target_sentences[:frequency_limit]
    return target_sentences

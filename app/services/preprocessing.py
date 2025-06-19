"""
preprocessing.py

Provides utilities to clean raw text and extract the top-N most frequent nouns
using NLTK's tokenizer and POS tagger.
"""

import re
from nltk import word_tokenize, pos_tag, FreqDist


def _basic_clean_text(text: str) -> str:
    """
    Standardize and clean raw text by:
    - Lowercasing
    - Removing punctuation
    - Normalizing whitespace

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and normalized text.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)      # Normalize whitespace
    return text.strip()

def _clean_token(token: str) -> str:
    """
    Clean a single word token: lowercase, remove punctuation.
    """
    return re.sub(r"[^\w\s]", "", token.lower())

def extract_top_n_nouns_with_frequency(
    text: str,
    top_n: int,
    stop_words: set,
    all_nouns: set
) -> dict[str, int]:
    """
    Extract top-N frequent nouns from text using preloaded stop words and noun sets.

    Args:
        text (str): Input raw text.
        top_n (int): Number of top nouns to return.
        stop_words (set): Preloaded set of stopwords.
        all_nouns (set): Preloaded set of valid nouns (e.g., WordNet nouns).

    Returns:
        dict[str, int]: Dictionary of nouns and their frequency.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")

    # Tokenize raw text
    tokens = word_tokenize(text)

    # POS tagging before cleaning
    tagged_tokens = pos_tag(tokens)

    # Now clean and lowercase nouns only
    nouns = []
    for word, tag in tagged_tokens:
        clean_word = _clean_token(word)
        if (
            tag.startswith("NN")
            and len(clean_word) > 1
            and clean_word.isalpha()
            and clean_word not in stop_words
            and clean_word in all_nouns
        ):
            nouns.append(clean_word)

    freq_dist = FreqDist(nouns)
    most_common = freq_dist.most_common(top_n)

    return dict(most_common)

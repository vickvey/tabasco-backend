"""
preprocessing.py

Provides utilities to clean raw text and extract the top-N most frequent nouns
using NLTK's tokenizer and POS tagger.
"""

import re
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords


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


def extract_top_n_nouns_with_frequency(text: str, top_n: int = 50) -> dict[str, int]:
    """
    Extract the top-N most frequent nouns from the input text.

    Uses NLTK for:
    - Tokenizing and cleaning the text
    - Filtering out stopwords
    - Part-of-speech tagging
    - Frequency analysis of nouns

    Args:
        text (str): Input text to analyze.
        top_n (int, optional): Number of top nouns to return. Defaults to 50.

    Raises:
        ValueError: If the input text is empty or not a string.

    Returns:
        dict[str, int]: A dictionary of the top `top_n` nouns and their frequencies.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")

    # Clean and tokenize
    text = _basic_clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Frequency distribution
    freq_dist = FreqDist(tokens)

    # POS tagging
    tagged_tokens = pos_tag(list(freq_dist.keys()))
    nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]

    # Sort by frequency
    sorted_nouns = sorted(nouns, key=lambda x: freq_dist[x], reverse=True)

    return {noun: freq_dist[noun] for noun in sorted_nouns[:top_n]}

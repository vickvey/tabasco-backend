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

    # POS tagging on full token list (in context)
    tagged_tokens = pos_tag(tokens)

    # Filter nouns only
    nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]

    # Frequency distribution of nouns only
    freq_dist = FreqDist(nouns)

    """OLD DEPRECATED
    # POS tagging
    # tagged_tokens = pos_tag(list(freq_dist.keys())) 
    # nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]
    """

    # Sort and get top N
    sorted_nouns = freq_dist.most_common(top_n)

    return dict(sorted_nouns)
    # return {noun: freq_dist[noun] for noun in sorted_nouns[:top_n]}

def extract_top_n_nouns_with_frequency_v2(
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

    cleaned_text = _basic_clean_text(text)
    tokens = word_tokenize(cleaned_text)
    tokens = [t for t in tokens if t.isalpha()]

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    tagged_tokens = pos_tag(tokens)

    # Filter nouns that are in the all_nouns set and longer than one character
    nouns = [
        word for word, tag in tagged_tokens
        if tag.startswith("NN") and len(word) > 1 and word.lower() in all_nouns
    ]

    freq_dist = FreqDist(nouns)
    most_common = freq_dist.most_common(top_n)

    return dict(most_common)
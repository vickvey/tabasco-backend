"""
sentence_processing.py

Utilities for extracting sentences from a block of text that contain a specific target word,
while ensuring compatibility with BERT tokenizer input limits. Sentences are cleaned,
filtered, and truncated if necessary to fit within token length constraints.
"""

import re
from functools import lru_cache
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer


@lru_cache(maxsize=1)
def _get_tokenizer():
    """
    Load and cache the BERT tokenizer (bert-base-uncased) for efficient reuse.

    Returns:
        BertTokenizer: Tokenizer instance for BERT.
    """
    return BertTokenizer.from_pretrained("bert-base-uncased")


def _standardize_text(text: str) -> str:
    """
    Cleans and standardizes raw text for downstream NLP tasks.

    This function performs the following steps:
    - Returns an empty string if the input is not a string.
    - Removes URLs (e.g., http://..., www...).
    - Removes email mentions (e.g., @username).
    - Removes special characters, symbols, and numeric digits.
    - Converts text to lowercase.
    - Removes isolated single-character tokens (e.g., 'a', 'b').
    - Collapses multiple consecutive whitespace characters into a single space.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The standardized and cleaned version of the input text.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", " ", text)      # Remove URLs
    text = re.sub(r"@\w+", " ", text)                # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", " ", text)         # Remove special chars and digits
    text = re.sub(r"\d+", " ", text)                 # Remove numbers
    text = text.lower()                              # Lowercase
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)        # Remove single characters
    text = re.sub(r"\s+", " ", text)                 # Normalize spaces

    return text.strip()


def _filter_sentences(sentences_list: list[str]) -> list[str]:
    """
    Applies text standardization to a list of sentences.

    Each sentence is cleaned using the `_standardize_text` function
    and returned in the same order.

    Args:
        sentences_list (list[str]): A list of raw sentence strings.

    Returns:
        list[str]: A list of standardized (cleaned) sentences.
    """
    return [_standardize_text(sentence) for sentence in sentences_list]


def get_sentences_with_target_word(
    text_content: str,
    target_word: str,
    frequency_limit: int = 100,
    max_length: int = 510
) -> list[str]:
    """
    Extract sentences from text that contain a specific target word.

    Sentences are:
    - Identified by checking if they contain the `target_word` (case-insensitive, exact match).
    - Filtered to ensure tokenized form fits within `max_length` tokens (BERT limit).
    - Truncated around the target word if they are too long.
    - Standardized using `_standardize_text` before being returned.

    Args:
        text_content (str): The input text from which to extract sentences.
        target_word (str): The word to search for in the sentences.
        frequency_limit (int, optional): Maximum number of sentences to return. Defaults to 100.
        max_length (int, optional): Maximum token length for BERT input. Defaults to 510.

    Returns:
        list[str]: A list of filtered, truncated, and cleaned sentences containing the target word.
    """
    tokenizer = _get_tokenizer()

    # Normalize and clean raw text
    text = " ".join(text_content.split()).replace("\n", " ").replace("\r", " ")

    # Split into individual sentences
    sentences = sent_tokenize(text)
    target_sentences = []

    for sent in sentences:
        if re.search(rf'\b{re.escape(target_word.lower())}\b', sent.lower()):
            tokens = tokenizer.tokenize(f"{tokenizer.cls_token} {sent} {tokenizer.sep_token}")
            if len(tokens) <= max_length:
                target_sentences.append(sent)
            else:
                # Attempt to truncate around the target word
                words = sent.split()
                target_idx = next((i for i, w in enumerate(words) if w.lower() == target_word.lower()), None)
                if target_idx is not None:
                    window_size = (max_length - 10) // 2
                    start = max(0, target_idx - window_size)
                    end = min(len(words), target_idx + window_size + 1)
                    truncated_sent = " ".join(words[start:end])
                    tokens = tokenizer.tokenize(f"{tokenizer.cls_token} {truncated_sent} {tokenizer.sep_token}")
                    if len(tokens) <= max_length and target_word.lower() in truncated_sent.lower():
                        target_sentences.append(truncated_sent)
                        print(f"Truncated sentence to {len(tokens)} tokens: {truncated_sent[:100]}...")

    return _filter_sentences(target_sentences[:frequency_limit])

import re
from typing import Dict
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords

class TextPreprocessor:

    @staticmethod
    def basic_clean_text(text: str) -> str:
        """
        Standardize and clean raw text by removing punctuation, normalizing whitespace and lowercasing.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        return text.strip()

    @staticmethod
    def extract_top_n_nouns_with_frequency(text: str, top_n: int = 50) -> Dict[str, int]:
        """
        Extract top-N most frequent nouns from given text.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string.")

        text = TextPreprocessor.basic_clean_text(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]

        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

        freq_dist = FreqDist(tokens)
        tagged_tokens = pos_tag(list(freq_dist.keys()))
        nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]

        sorted_nouns = sorted(nouns, key=lambda x: freq_dist[x], reverse=True)
        return {noun: freq_dist[noun] for noun in sorted_nouns[:top_n]}

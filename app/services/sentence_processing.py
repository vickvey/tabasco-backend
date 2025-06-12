# app/services/sentence_processing.py
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from functools import lru_cache


class SentenceProcessor:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_tokenizer():
        return BertTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def process_sentences(text: str, target_word, frequency_limit=100, max_length=510):
        # Initialize BERT tokenizer for length checking
        tokenizer = TextProcessingService.get_tokenizer()

        # Clean text: remove extra newlines, multiple spaces, and common PDF artifacts
        text = " ".join(text.split())  # Normalize whitespace
        text = text.replace("\n", " ").replace("\r", " ")  # Remove newlines

        # Split into sentences
        sentences = sent_tokenize(text)
        target_sentences = []

        for sent in sentences:
            # Check if target word is present (case-insensitive)
            if target_word.lower() in sent.lower():
                # Tokenize to check length
                tokens = tokenizer.tokenize(f"[CLS] {sent} [SEP]")
                if len(tokens) <= max_length:
                    target_sentences.append(sent)
                else:
                    # Split long sentence into smaller chunks around the target word
                    words = sent.split()
                    target_idx = next((i for i, w in enumerate(words) if w.lower() == target_word.lower()), None)
                    if target_idx is not None:
                        # Define a window to keep around the target word
                        window_size = (max_length - 10) // 2  # Approximate tokens per word
                        start = max(0, target_idx - window_size)
                        end = min(len(words), target_idx + window_size + 1)
                        truncated_sent = " ".join(words[start:end])
                        # Verify the truncated sentence still contains the target word
                        tokens = tokenizer.tokenize(f"[CLS] {truncated_sent} [SEP]")
                        if len(tokens) <= max_length and target_word.lower() in truncated_sent.lower():
                            target_sentences.append(truncated_sent)
                            print(f"Truncated sentence to {len(tokens)} tokens: {truncated_sent[:100]}...")

        # Limit the number of sentences
        return target_sentences[:frequency_limit]

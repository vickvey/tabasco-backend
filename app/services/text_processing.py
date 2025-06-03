# services/text_processing.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, FreqDist
from app.settings import UPLOAD_FOLDER
from app.utils.helpers import pdf2text
from transformers import BertTokenizer

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

def process_sentences(text, target_word, frequency_limit=100, max_length=510):
    # Download NLTK resources if not already present
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    # Initialize BERT tokenizer for length checking
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

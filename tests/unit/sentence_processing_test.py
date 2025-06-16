import pytest
from app.services.sentence_processing import get_sentences_with_target_word

sample_text = """
Machine learning is a subset of artificial intelligence. 
Deep learning is a branch of machine learning. 
The BERT model revolutionized NLP. 
In 2018, Google introduced BERT: Bidirectional Encoder Representations from Transformers. 
BERT has been pre-trained on large corpora. 
This sentence does not contain the keyword. 
Another line about bert, lowercase.
"""

def test_basic_target_word_detection():
    result = get_sentences_with_target_word(sample_text, "BERT")
    assert isinstance(result, list)
    assert all("bert" in s.lower() for s in result)
    assert len(result) > 0

def test_case_insensitive_matching():
    result_upper = get_sentences_with_target_word(sample_text, "BERT")
    result_lower = get_sentences_with_target_word(sample_text, "bert")
    assert result_upper == result_lower

def test_token_length_constraint():
    long_sentence = "BERT " + ("word " * 600)
    text = f"This is a normal sentence. {long_sentence}"
    result = get_sentences_with_target_word(text, "BERT", max_length=510)
    assert len(result) == 1
    assert "BERT" in result[0] or "bert" in result[0]
    assert len(result[0].split()) < 600  # it should be truncated

def test_frequency_limit():
    repeated_text = "BERT is great. " * 200
    result = get_sentences_with_target_word(repeated_text, "BERT", frequency_limit=10)
    assert len(result) == 10

def test_word_boundary():
    text = "Albert is a name. BERT is a model. Rebert is not relevant. The best model is BERT."
    result = get_sentences_with_target_word(text, "BERT")
    assert all("bert" in s.lower() for s in result)
    assert all("rebert" not in s.lower() and "albert" not in s.lower() for s in result)
    assert len(result) == 2  # Only exact matches of 'bert'

def test_no_match():
    text = "This sentence does not contain the target."
    result = get_sentences_with_target_word(text, "bert")
    assert result == []

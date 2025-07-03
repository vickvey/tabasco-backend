import nltk

def ensure_nltk_data():
    for resource in ['punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
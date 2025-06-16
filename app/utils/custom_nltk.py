import nltk

def ensure_nltk_data():
    for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
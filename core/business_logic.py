import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, FreqDist
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

# Ensure necessary NLTK packages are downloaded.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def get_noun_list(text, top_n=50):
    """
    Tokenize text and return the top N nouns based on frequency.
    """
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

def process_sentences(text, target_word, frequency_limit=100):
    """
    Split text into sentences and return those containing the target word.
    Optionally limit the number of sentences to process.
    """
    sentences = sent_tokenize(text)
    target_sentences = [sent for sent in sentences if target_word.lower() in sent.lower()]
    if len(target_sentences) > frequency_limit:
        target_sentences = target_sentences[:frequency_limit]
    return target_sentences

import torch.nn.functional as F

class DisambModel:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_sentence):
        marked_text = "[CLS] " + input_sentence + " [SEP]"
        tokens = self.tokenizer.tokenize(marked_text)

        # Truncate if too long
        if len(tokens) > 512:
            tokens = tokens[:512]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)

        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensor = torch.tensor([segments_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensor)
            hidden_states = outputs.hidden_states

        token_embeddings = []
        for i in range(len(tokens)):
            layers = [hidden_states[-j][0][i] for j in range(1, 5)]
            token_embedding = torch.sum(torch.stack(layers), dim=0)
            token_embeddings.append(token_embedding)

        return tokens, token_embeddings

    def get_context_words(self, sentence, target_word, top_k=10):
        tokens, embeddings = self.forward(sentence)
        target_word = target_word.lower()

        # Get positions of the target word in the token list
        target_indices = [i for i, token in enumerate(tokens) if target_word in token]
        if not target_indices:
            return [], []

        target_vector = embeddings[target_indices[0]]

        similarities = []
        for i, (token, vector) in enumerate(zip(tokens, embeddings)):
            if i == target_indices[0] or token in ["[CLS]", "[SEP]"]:
                continue
            similarity = F.cosine_similarity(target_vector, vector, dim=0).item()
            similarities.append((token, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        top_similar = similarities[:top_k]
        return top_similar


def get_target_matrix(sentences, disamb_model):
    """
    For each sentence compute an embedding vector.
    Then construct a cosine-similarity matrix between all sentence embeddings.
    """
    embeddings = []
    for sent in sentences:
        tokens, token_embs = disamb_model.forward(sent)
        # For simplicity, we compute a sentence embedding as the average of token embeddings.
        sent_emb = torch.mean(torch.stack(token_embs), dim=0)
        embeddings.append(sent_emb)
    n = len(embeddings)
    matrix = torch.zeros((n, n))
    cos = torch.nn.CosineSimilarity(dim=0)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cos(embeddings[i], embeddings[j])
    return matrix

def get_clusters(matrix, num_clusters):
    """
    Cluster the sentences using KMeans based on the computed similarity matrix.
    """
    X = matrix.numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels.tolist()

def label_sentences_by_cluster(sentences, matrix, num_clusters):
    """
    Labels sentences by cluster using cosine similarity matrix and returns a dict {cluster_id: [sentences]}.
    """
    labels = get_clusters(matrix, num_clusters)
    cluster_dict = {}
    for sent, label in zip(sentences, labels):
        cluster_dict.setdefault(label, []).append(sent)
    return cluster_dict

import random
import operator

def generate_summary_files(target_word, clusters_dict, summary_folder_path):
    """
    Write summary files for each cluster showing related context words and representative sentences.
    """
    for cluster_num, sentences in clusters_dict.items():
        word_freq = {}
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            for word in tokens:
                if word.lower() != target_word.lower() and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in sorted_words[:50]]

        file_path = summary_folder_path / f"summary_text_{cluster_num}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Top-most context words from the cluster:\n")
            f.write("***********************************************\n")
            f.write(" , ".join(top_words) + "\n")
            f.write("***********************************************\n\n")

            for idx, sentence in enumerate(sentences):
                f.write(f"{target_word.title()} {idx+1}\n")
                f.write(f"Instance {idx+1} of {target_word.title()} belongs to Cluster {int(cluster_num)+1}\n")
                f.write("\n~~~~~~\n")
                f.write("Corresponding Sentence:\n")
                f.write(sentence.strip())
                f.write("\n")
                f.write("##############################################################\n\n")



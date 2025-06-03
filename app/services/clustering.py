# services/clustering.py
import torch
from sklearn.cluster import KMeans
from kneed import KneeLocator


def get_target_matrix(sentences, disamb_model, target_word):
    """
    For each sentence compute an embedding vector.
    Then construct a cosine-similarity matrix between all sentence embeddings.
    """
    embeddings = []
    for sent in sentences:
        tokens = disamb_model.tokenizer.tokenize(f"[CLS] {sent} [SEP]")
        if len(tokens) > 512:
            print(f"Warning: Sentence exceeds 512 tokens (length={len(tokens)}): {sent}")
        target_emb = disamb_model.forward(sent, target_word)
        embeddings.append(target_emb)
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

def suggest_num_clusters(matrix):
    # Implementation from business_logic.py
    X = matrix.numpy()
    wcss = []
    for i in range(1, min(11, X.shape[0] + 1)):
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    if len(wcss) < 2:
        return 1
    kn = KneeLocator(range(1, len(wcss) + 1), wcss, curve='convex', direction='decreasing')
    return kn.knee if kn.knee else 1

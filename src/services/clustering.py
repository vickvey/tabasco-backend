# services/clustering.py
import torch
from sklearn.cluster import KMeans
from kneed import KneeLocator

def get_target_matrix(sentences, disamb_model, target_word):
    # Implementation for generating target matrix
    pass

def label_sentences_by_cluster(sentences, matrix, num_clusters):
    # Implementation for clustering sentences
    pass

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
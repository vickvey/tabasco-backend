from sklearn.cluster import KMeans
from kneed import KneeLocator

def get_clusters(matrix, num_clusters):
    X = matrix.numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans.fit_predict(X).tolist()

def label_sentences_by_cluster(sentences, matrix, num_clusters):
    labels = get_clusters(matrix, num_clusters)
    cluster_dict = {}
    for sent, label in zip(sentences, labels):
        cluster_dict.setdefault(label, []).append(sent)
    return cluster_dict

def suggest_num_clusters(matrix):
    X = matrix.numpy()
    wcss = []
    for i in range(1, min(11, X.shape[0] + 1)):
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    if len(wcss) < 2:
        return 1
    kn = KneeLocator(range(1, len(wcss) + 1), wcss, curve='convex', direction='decreasing')
    return kn.knee or 1

def suggest_num_clusters_with_data(matrix):
    """
    Suggest optimal number of clusters using the elbow method and also return data
    to plot the elbow curve in the frontend.
    
    Returns:
        - optimal_k: suggested number of clusters
        - k_range: list of k values tested
        - wcss: corresponding WCSS values
    """
    X = matrix.numpy()
    wcss = []
    k_range = list(range(1, min(11, X.shape[0] + 1)))  # from 1 to max 10 or data size

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    if len(wcss) < 2:
        return 1, k_range, wcss

    kn = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
    return kn.knee or 1, k_range, wcss

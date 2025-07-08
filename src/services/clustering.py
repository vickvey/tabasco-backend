import torch
from sklearn.cluster import KMeans
from kneed import KneeLocator

# TODO: Look for a PCA in original app.py in get_clusters function [Might be something]

def suggest_num_clusters_with_data(matrix):
    """
    Suggest the optimal number of clusters using the Elbow Method.
    Also returns the data needed to plot the elbow curve in the frontend.

    Parameters:
        matrix (torch.Tensor): A 2D tensor (sentence embeddings) representing data points.

    Returns:
        optimal_k (int): The estimated best number of clusters using KneeLocator.
        k_range (List[int]): List of k values (number of clusters) tested.
        wcss (List[float]): Within-cluster sum of squares for each k (used to plot the elbow curve).
    """
    # Convert the tensor to a NumPy array for compatibility with scikit-learn
    X = matrix.numpy()
    wcss = []  # List to store Within-Cluster Sum of Squares for each value of k
    # Define range of k values (number of clusters) to test. Max is min(10, number of samples)
    k_range = list(range(1, min(11, X.shape[0] + 1)))

    # Loop over each k and compute WCSS using KMeans
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",      # Smart centroid initialization
            max_iter=300,          # Max number of iterations per run
            n_init=10,             # Number of different centroid seeds to try
            random_state=42        # Reproducible results
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS

    # If we have fewer than 2 points, clustering doesn't make sense
    if len(wcss) < 2:
        return 1, k_range, wcss

    # Use KneeLocator to find the "elbow" point where adding more clusters doesn't help much
    kn = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')

    # Return the detected knee (optimal number of clusters) or 1 as fallback
    return kn.knee or 1, k_range, wcss


def _get_clusters(matrix, num_clusters):
    """
    Run KMeans clustering on the matrix with a fixed number of clusters.

    Parameters:
        matrix (torch.Tensor): The input 2D tensor of embeddings.
        num_clusters (int): The number of clusters to form.

    Returns:
        List[int]: Cluster labels for each data point.
    """
    X = matrix.numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)  # Predict the cluster each sample belongs to
    return labels.tolist()


def label_sentences_by_cluster(sentences: list[str], matrix: torch.Tensor, num_clusters: int) -> dict[int, list[str]]:
    labels = _get_clusters(matrix, num_clusters)
    cluster_dict = {}
    for sent, label in zip(sentences, labels):
        cluster_dict.setdefault(label, []).append(sent)
    return cluster_dict

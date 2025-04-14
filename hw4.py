
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.metrics import silhouette_score, rand_score
import pandas as pd

# Load the dataset
data = pd.read_csv('spiral-dataset.csv', sep='\t', header=None)
X = data.iloc[:, :2].values  # X and Y coordinates
true_labels = data.iloc[:, 2].values  # True cluster labels

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def linkage_distance(cluster1, cluster2, X, method="single"):
    if method == "single":
        return np.min([euclidean(X[i], X[j]) for i in cluster1 for j in cluster2])
    elif method == "complete":
        return np.max([euclidean(X[i], X[j]) for i in cluster1 for j in cluster2])
    elif method == "average":
        return np.mean([euclidean(X[i], X[j]) for i in cluster1 for j in cluster2])
    elif method == "centroid":
        c1 = np.mean(X[cluster1], axis=0)
        c2 = np.mean(X[cluster2], axis=0)
        return euclidean(c1, c2)
    else:
        raise ValueError("Unknown method.")
    
def hierarchical_clustering(X, num_clusters=3, method="single"):
    clusters = [[i] for i in range(len(X))]
    distances = {}

    while len(clusters) > num_clusters:
        min_dist = float("inf")
        to_merge = (0, 1)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                key = (tuple(c1), tuple(c2))
                if key not in distances:
                    distances[key] = linkage_distance(c1, c2, X, method)
                dist = distances[key]
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)

        i, j = to_merge
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    # Generate label assignments
    labels = np.zeros(len(X), dtype=int)
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            labels[point] = idx
    return labels

def compute_sse(X, labels):
    sse = 0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        centroid = cluster_points.mean(axis=0)
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse

def linkage_matrix(X, method="single"):
    return linkage(X, method=method)

def evaluate_clustering(X, labels, true_labels, method):
    sse = compute_sse(X, labels)
    ri = rand_score(true_labels, labels)
    sil = silhouette_score(X, labels)

    try:
        Y = pdist(X)
        Z = linkage_matrix(X, method=method)
        coph_corr, _ = cophenet(Z, Y)
    except ValueError:
        coph_corr = np.nan  # Not supported for centroid

    return sse, ri, coph_corr, sil

def plot_clusters(X, labels, method):
    plt.figure(figsize=(6, 5))
    for k in np.unique(labels):
        cluster = X[labels == k]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {k}")
    plt.title(f"Clustering result - {method} linkage")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main evaluation
methods = ["single", "complete", "average", "centroid"]
results = {}

for method in methods:
    print(f"\nMethod: {method.capitalize()} Linkage")
    labels = hierarchical_clustering(X, num_clusters=3, method=method)
    sse, ri, coph_corr, sil = evaluate_clustering(X, labels, true_labels, method)
    results[method] = (sse, ri, coph_corr, sil)
    print(f"SSE: {sse:.2f}, Rand Index: {ri:.3f}, Cophenetic Corr.: {coph_corr:.3f}, Silhouette: {sil:.3f}")
    plot_clusters(X, labels, method)

print("\nSummary of Results:")
for method in results:
    sse, ri, coph, sil = results[method]
    print(f"{method.capitalize()} Linkage => SSE: {sse:.2f}, RI: {ri:.3f}, CCC: {coph:.3f}, Sil: {sil:.3f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean, cityblock, cosine

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=123)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='red')
plt.title('KMeans Clustering')
plt.show()

# Perform hierarchical clustering
linked = linkage(X, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# Sample points
point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])

# Calculate distances
euclidean_distance = euclidean(point1, point2)
manhattan_distance = cityblock(point1, point2)
cosine_sim = 1 - cosine(point1, point2)

print(f"Euclidean Distance: {euclidean_distance:.2f}")
print(f"Manhattan Distance: {manhattan_distance:.2f}")
print(f"Cosine Similarity: {cosine_sim:.2f}")

from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"The average silhouette score is: {silhouette_avg:.3f}")

# Interpret the score:
if silhouette_avg > 0.5:
    print("The clustering shows a strong structure.")
elif silhouette_avg > 0.25:
    print("The clustering shows a reasonable structure.")
else:
    print("The clustering structure is weak and could be artificial.")
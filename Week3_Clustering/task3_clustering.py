import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Dataset Load Karein (Iris Dataset for Clustering)
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Select features for 2D Clustering visualization
X = df[["sepal length (cm)", "sepal width (cm)"]]

# 2. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow Method to find optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 4. Apply K-Means with K=3
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

# 5. Visualizations
plt.figure(figsize=(12, 5))

# Plot 1: Elbow Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss, marker="o", color="blue")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")

# Plot 2: K-Means Clusters
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    hue=clusters,
    palette="Set1",
    s=100,
    style=clusters,
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=250,
    c="yellow",
    marker="X",
    label="Centroids",
)
plt.title("K-Means Clustering Result (K=3)")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.legend()

plt.tight_layout()
plt.savefig("clustering_visualizations.png")
plt.show()

print("Clustering execution finished! Saved as 'clustering_visualizations.png'")

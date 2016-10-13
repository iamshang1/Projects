import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

# generate gaussian-distributed random data
X,y = make_blobs(n_samples=2000,n_features=2,centers=10,random_state=1)
plt.scatter(X[:,0],X[:,1])
plt.title("Scatter Plot of Randomly Generated Data")
plt.savefig('scatter1.png')
plt.show()

# formula to calculate explained variance for K-means
def explained_variance(cluster,data):
    center = data.mean(axis=0)
    SST = np.sum((data - center)**2)
    SSE = 0
    for i in range(len(cluster.cluster_centers_)):
        mask = cluster.labels_ == i
        cluster_points = data[mask]
        SSE += np.sum((cluster_points - cluster.cluster_centers_[i])**2)
    return 1 - (SSE/SST)

# try different K for K-means    
r2 = []
silhouette = []
for i in range(3,20):
    cluster = KMeans(n_clusters=i)
    cluster.fit(X)
    r2.append(explained_variance(cluster,X))
    silhouette.append(silhouette_score(X, cluster.predict(X)))

# plot explained variance for different Ks
plt.plot(range(3,20), r2)
plt.title("Explained Variance for Different K-Means")
plt.xlabel('Number of Centroids')
plt.ylabel('Explained Variance')
plt.savefig('kmeans.png')
plt.show()

# plot silhouette score for different Ks
plt.plot(range(3,20), silhouette)
plt.title("Silhouette Score for Different K-Means")
plt.xlabel('Number of Centroids')
plt.ylabel('Silhouette Score')
plt.savefig('silhouette.png')
plt.show()

# scatter plot of data separated into 7 clusters
colors = ['r','g','b','y','orange','indigo','magenta']
for i in range(7):
    mask = best.labels_ == i
    cluster_points = X[mask]
    plt.scatter(cluster_points[:,0],cluster_points[:,1],c=colors[i])
plt.title("Scatter Plot of Data Separated into 7 Clusters")
plt.savefig('scatter2.png')
plt.show()
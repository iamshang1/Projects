# K-Means Clustering

This exercise uses the K-Means clustering algorithm to cluster randomly generated gaussian blobs into clusters.

Different values of K are used cluster the data. The following metrics are used to evaluate the effectiveness of each K:

 - Explained Variance, which is a measure based on the distance of each point from the nearest centroid
 - Silhouette Score, which is a measure based on the amount of overlap between clusters
 
## Results

![scatter1](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Kmeans_Clustering/scatter1.png)

![kmeans](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Kmeans_Clustering/kmeans.png)

![silhouette](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Kmeans_Clustering/silhouette.png)

## Conclusion

Based on the two graphs above, we see that 7 clusters has a high explained variance and is a good "elbow point" when using the
elbow method. 7 clusters also has a good silhouette score. Based on these two metrics we decide that 7 clusters is a good balance between 
fitting the data while not overlapping the clusters.

![scatter2](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Kmeans_Clustering/scatter2.png)
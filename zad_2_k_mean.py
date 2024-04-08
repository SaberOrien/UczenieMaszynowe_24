from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import numpy as np
import time

# Load data
wine = load_wine()
X = wine.data

# KFold setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Possible values for k (number of clusters)
cluster_values = [2, 3, 4, 5, 6, 7]

print("k, Avg Silhouette Score, Avg Training Time (s), Total Execution Time (s)")

for k in cluster_values:
    training_times = []
    execution_times = []
    silhouette_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train)
        end_train_time = time.time()
        
        # Compute silhouette score on the test set
        labels_test = kmeans.predict(X_test)
        silhouette_avg = silhouette_score(X_test, labels_test)
        silhouette_scores.append(silhouette_avg)
        
        end_time = time.time()
        
        training_times.append(end_train_time - start_time)
        execution_times.append(end_time - start_time)
    
    print(f'{k}, {np.mean(silhouette_scores):.4f}, {np.mean(training_times):.4f}, {np.mean(execution_times):.4f}')

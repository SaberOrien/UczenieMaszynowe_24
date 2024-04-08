from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

wine = load_wine()
X, y = wine.data, wine.target

kf = KFold(n_splits=10, shuffle=True, random_state=42)

k_values = [1, 3, 5, 7]
distance_metrics = ['euclidean', 'manhattan', 'minkowski']

print("k, Metric, Avg Accuracy, Std Dev, Avg Training Time (s), Total Execution Time (s)")

for k in k_values:
    for metric in distance_metrics:
        training_times = []
        execution_times = []
        scores = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            
            start_train_time = time.time()
            clf.fit(X_train, y_train)
            end_train_time = time.time()
            training_time = end_train_time - start_train_time
            training_times.append(training_time)
            
            start_test_time = time.time()
            y_pred = clf.predict(X_test)
            end_test_time = time.time()
            execution_time = end_test_time - start_train_time
            execution_times.append(execution_time)
            
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
        
        avg_training_time = np.mean(training_times)
        avg_execution_time = np.mean(execution_times)
        scores = np.array(scores)
        
        print(f'{k}, {metric}, {scores.mean():.4f}, {scores.std():.4f}, {avg_training_time:.4f}, {avg_execution_time:.4f}')

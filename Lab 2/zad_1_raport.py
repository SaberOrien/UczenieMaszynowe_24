from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Funkcja do przypisywania etykiet klas na podstawie najczęstszej etykiety w każdym klastrze
def assign_labels_to_clusters(y_true, y_pred):
    from scipy.stats import mode
    labels = np.zeros_like(y_pred)
    for i in range(np.max(y_pred) + 1):
        mask = (y_pred == i)
        if np.any(mask):
            labels[mask] = mode(y_true[mask])[0]
    return labels

# Załadowanie zbioru danych Iris
iris = load_iris()
X, y = iris.data, iris.target

# Inicjalizacja walidacji krzyżowej
kf = KFold(n_splits=10, shuffle=True, random_state=42)

k_values = [1, 3, 5, 7]

print("k, Avg Accuracy, Std Dev, Avg Training Time (s), Total Execution Time (s)")

for k in k_values:
    training_times = []
    execution_times = []
    scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = KMeans(n_clusters=k, random_state=42)
        
        start_train_time = time.time()
        clf.fit(X_train)
        end_train_time = time.time()
        training_time = end_train_time - start_train_time
        training_times.append(training_time)
        
        start_test_time = time.time()
        y_pred = clf.predict(X_test)

        y_pred_labels = assign_labels_to_clusters(y_test, y_pred)
        end_test_time = time.time()
        execution_time = end_test_time - start_train_time
        execution_times.append(execution_time)
        
        accuracy = accuracy_score(y_test, y_pred_labels)
        scores.append(accuracy)
    
    avg_training_time = np.mean(training_times)
    avg_execution_time = np.mean(execution_times)
    scores = np.array(scores)
    
    print(f'{k}, {scores.mean():.4f}, {scores.std():.4f}, {avg_training_time:.4f}, {avg_execution_time:.4f}')

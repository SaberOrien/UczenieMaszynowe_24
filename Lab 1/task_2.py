from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = [3, 5, 7]
metrics = ['euclidean', 'manhattan']

for k in k_values:
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Dokładność dla k={k}, metryka={metric}: {accuracy:.4f}')
        
        cv_scores = cross_val_score(knn, X, y, cv=5)
        print(f'Średnia dokładność walidacji krzyżowej dla k={k}, metryka={metric}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = [3, 5, 7]
test_accuracies = []
cv_means = []
cv_stds = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(accuracy)
    print(f'Dokładność dla k={k}: {accuracy:.4f}')

    cv_scores = cross_val_score(knn, X, y, cv=5)
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())
    print(f'Średnia dokładność walidacji krzyżowej dla k={k}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n')

# Tworzenie wykresów
plt.figure(figsize=(10, 6))

# Dokładność na zbiorze testowym
plt.plot(k_values, test_accuracies, label='Dokładność na zbiorze testowym', marker='o', linestyle='-', color='blue')

# Średnia dokładność z walidacji krzyżowej
plt.errorbar(k_values, cv_means, yerr=cv_stds, label='Walidacja krzyżowa', marker='s', linestyle='--', color='red')

plt.title('Dokładność modelu kNN w zależności od liczby sąsiadów k')
plt.xlabel('Liczba sąsiadów k')
plt.ylabel('Dokładność')
plt.legend()
plt.xticks(k_values)
plt.show()
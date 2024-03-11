from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
import numpy as np

# Wczytanie zbioru danych Iris i ograniczenie do pierwszych dwóch cech
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Definicja kolorów i mapy kolorów
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

# Utworzenie modelu kNN
knn = KNeighborsClassifier(n_neighbors=3)

# Utworzenie modelu NCA i kNN w pipeline
nca = Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=42)), ('knn', KNeighborsClassifier(n_neighbors=3))])

# Trenowanie modeli
knn.fit(X_train, y_train)
nca.fit(X_train, y_train)

# Utworzenie siatki punktów do pokazania granic decyzyjnych
h = .02  # krok siatki
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Przewidywanie klas dla każdego punktu na siatce dla kNN
Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_knn = Z_knn.reshape(xx.shape)

# Przewidywanie klas dla każdego punktu na siatce dla NCA
Z_nca = nca.predict(np.c_[xx.ravel(), yy.ravel()])
Z_nca = Z_nca.reshape(xx.shape)

# Rysowanie granic decyzyjnych dla kNN
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, Z_knn, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Granice decyzyjne kNN (3-NN)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Rysowanie granic decyzyjnych dla NCA
plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, Z_nca, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Granice decyzyjne NCA (3-NN)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()
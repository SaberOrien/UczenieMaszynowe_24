from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

wine = load_wine()
# Wybór tylko dwóch cech dla celów wizualizacji
X = wine.data[:, :2]
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definicja modelu kNN
k = 3  # Możesz zmienić na inne wartości k
metric = 'euclidean'  # Możesz zmienić na 'manhattan'
knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
knn.fit(X_train, y_train)

# Definicja kolorów i mapy kolorów
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Utworzenie siatki punktów do pokazania granic decyzyjnych
h = .02  # krok siatki
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Przewidywanie klas dla każdego punktu na siatce
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Rysowanie granic decyzyjnych
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Dodanie punktów treningowych
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"3-Class classification (k = {k}, metric = {metric})")
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])

plt.show()

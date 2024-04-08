from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
from graphviz import render

# Funkcja do generowania wykresu granic decyzyjnych
def plot_decision_boundaries(X, y, classifier, resolution=0.02):
    # Konfiguracja markerów i mapy kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu

    # Rysowanie granic decyzyjnych
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Rysowanie próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=iris.target_names[cl], edgecolor='black')

iris = load_iris()
X = iris.data[:, :2] 
y = iris.target

clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
clf.fit(X, y)

# 1. Bazując na cechach Iris i korzystając z interfejsu scikit-learn, 
# stwórz drzewo decyzyjne o maksymalnej wysokości 3, 
# a na kryterium zanieczyszczenia dobierz współczynnik Gini.
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.show()

# 2. Wygeneruj wykres granic decyzyjnych.
plt.figure(figsize=(20, 10))
plot_decision_boundaries(X, y, classifier=clf)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(loc='upper left')
plt.show()

# 3. Wykorzystując możliwość zapisu drzewa w rozszerzeniu .dot 
# i programu Graphviz, wygeneruj wykres drzewa. 

# Eksportuj drzewo do pliku .dot
export_graphviz(clf, out_file='tree.dot', 
                feature_names=iris.feature_names[:2],
                class_names=iris.target_names,
                rounded=True, proportion=False, 
                precision=2, filled=True)

render('dot', 'png', 'tree.dot')

# 4. Wykonaj to samo zadanie w programie WEKA.

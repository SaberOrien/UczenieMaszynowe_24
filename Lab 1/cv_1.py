from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = NearestCentroid()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)

scores = np.array(scores)

print(f'Accuracy scores for the Iris dataset: {scores}')
print(f'Average accuracy: {scores.mean():.4f}, Standard Deviation: {scores.std():.4f}')

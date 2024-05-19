from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import xgboost as xgb
import time

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Wyniki - dictionary
results = {}

# RandomForest
max_depths = [None, 10, 20, 30]
min_samples_splits = [2, 4, 6, 8]

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        rf.fit(X_train, y_train)
        end_time = time.time()
        rf_predictions = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        results[(f"RF max_depth={max_depth}", f"min_samples_split={min_samples_split}")] = (rf_accuracy, end_time - start_time)

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

max_depths = [3, 5, 7]
subsamples = [0.7, 0.8, 0.9]

for max_depth in max_depths:
    for subsample in subsamples:
        start_time = time.time()
        params = {
            'eta': 0.1,
            'max_depth': max_depth,
            'subsample': subsample,
            'objective': 'binary:logistic'
        }
        bst = xgb.train(params, dtrain, num_boost_round=100)
        bst_predictions = bst.predict(dtest)
        bst_predictions = np.round(bst_predictions)
        bst_accuracy = accuracy_score(y_test, bst_predictions)
        end_time = time.time()
        results[(f"XGB max_depth={max_depth}", f"subsample={subsample}")] = (bst_accuracy, end_time - start_time)

# wyniki
for params, acc_time in results.items():
    print(f"Params: {params[0]}, {params[1]} - Accuracy: {acc_time[0]:.2f}, Time: {acc_time[1]:.2f} s")
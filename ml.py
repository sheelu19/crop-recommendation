import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings("ignore")
df = pd.read_csv("crop_prediction_model_one.csv")
X = df.drop("label", axis=1)
y = df["label"]
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.2, random_state=42))
grid = {"n_estimators": [i for i in range(200, 1200, 10)],
        "max_depth": [i for i in range(1, 30)],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [i for i in range(2, 6)],
        "min_samples_leaf": [i for i in range(1, 6)]}
clf = RandomForestClassifier(n_jobs=1)
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=10,
                            cv=5,
                            verbose=2)
rs_clf.fit(X_train, y_train)
params = rs_clf.best_params_
n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth = itemgetter(
     "n_estimators",
     "min_samples_split",
     "min_samples_leaf",
     "max_features",
     "max_depth"
 )(params)
rfc = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, max_features=max_features,
                             max_depth=max_depth)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
pickle.dump(rfc, open('model.pkl', 'wb'))


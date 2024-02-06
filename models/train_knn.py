# basics
import pandas as pd

# models
from sklearn.neighbors import KNeighborsClassifier

# accuracy metrics
from sklearn.metrics import accuracy_score

# model deployment
import pickle

data = pd.read_csv("../data/ACME-HappinessSurvey2020.csv")

# Prepare data
features_to_remove = ['X2', 'X4']
data_selected = data.drop(features_to_remove, axis=1)
X = data_selected.drop(['Y'], axis=1)
y = data_selected["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


model_knn = KNeighborsClassifier(n_neighbors=20)
model_knn.fit(X_train, y_train)

# Evaluation
y_pred = model_knn.predict(X_test)
accuracy_score(y_test, y_pred)


with open('trained_knn_model.pkl', 'wb') as f:
    pickle.dump(model_knn, f)
import pickle
import pandas as pd
with open('trained_knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)


def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions

new_data = pd.read_csv("../data/ACME-HappinessSurvey2020.csv")
y = new_data['Y']
new_data.drop('Y', axis=1, inplace=True)

predictions = predict(knn, new_data)
predictions
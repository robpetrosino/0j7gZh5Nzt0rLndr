# basics
import pandas as pd

# model deployment
import pickle

with open('trained_model.pkl', 'rb') as f:
    model_dt = pickle.load(f)

def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions

new_data = pd.read_csv("../data/ACME-HappinessSurvey2020.csv")
features_to_remove = ['X2', 'X4', 'X5']
new_data_sel = new_data.drop(features_to_remove, axis=1)
y = new_data['Y']
new_data.drop('Y', axis=1, inplace=True)

predictions = predict(model_dt, new_data)
predictions
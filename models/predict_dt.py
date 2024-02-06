import pickle
import pandas as pd
with open('trained_dt_model.pkl', 'rb') as f:
    dt = pickle.load(f)

def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions

new_data = pd.read_csv("../data/ACME-HappinessSurvey2020.csv")
features_to_remove = ['X2', 'X4', 'X5']
y = new_data['Y']
new_data.drop(features_to_remove+['Y'], axis=1, inplace=True)

predictions = predict(dt, new_data)
predictions
This repository contains the code for a machine learning model that is trained on the ACME Happiness Survey dataset provided by Apziva. The goal of the model is to predict customer satisfaction based on survey responses.

## Prerequisites

The following packages are required to run the code:

1. `pandas` 
2. `numpy`
3. `matplotlib` and `seaborn`
4. `scikit-learn`

You can install these packages by running the following command: `pip install -r requirements.txt`

# Train and predict

I found that both the KNN and the Decision Tree classifying algorithms perform equally well in predicting the target label. The code for the KNN model can be found in the `train_knn.py` file. The code for the Decision Tree model can be found in the `train_dt.py` file. To train the model, run the following command: `python [file]`

The model will be trained on the `ACME-HappinessSurvey2020.csv` dataset and the accuracy will be printed.

Similarly the code for using the trained model to make predictions on new data is contained in the `predict_knn.py` and `predict_dt.py` for the KNN and Decision Tree classifing algorithms, respectively.

# Evaluation

The model uses  algorithm with `accuracy_score` and `f1_score` as evaluation metric.

# Conclusion

In this project, I leveraged the power of machine learning and employed the decision tree recursive algorithm to construct a robust, feature-efficient, and highly accurate ML classifier. The main goal was to predict the customers' satisfaction of the services provided. The model achieves an accuracy of 74% with both the KNN and Decision Tree models.

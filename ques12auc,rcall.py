# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:54:35 2024

12.	Perform evaluation metrics for all algorithms
a.	AUC
b.	Precision
c.	Recall
d.	Specificity
e.	Sensitivity
f.	Mean absolute percentage error 

@author: vasit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, mean_absolute_error
from sklearn.metrics import precision_recall_curve

# Load the dataset
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8, 4, 10, 10, 1, 5, 7, 0, 7, 1, 1, 3],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 110, 168, 139, 189, 166, 100, 118, 107, 103, 115, 126],
    'BP': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96, 92, 74, 80, 60, 72, 0, 84, 74, 30, 70, 88],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0, 0, 0, 0, 23, 19, 0, 47, 0, 38, 30, 41],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0, 0, 0, 0, 846, 175, 0, 230, 0, 83, 96, 235],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31, 35.3, 30.5, 30.5, 37.6, 38, 27.1, 30.1, 25.8, 30, 45.8, 29.6, 43.3, 34.6, 39.3],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232, 0.191, 0.537, 1.441, 0.398, 0.587, 0.484, 0.551, 0.254, 0.183, 0.529, 0.704],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 30, 34, 57, 59, 51, 32, 31, 31, 33, 32, 27],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

# Train both models
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Predict on test data
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Calculate the evaluation metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)
rf_specificity = rf_conf_matrix[0][0] / (rf_conf_matrix[0][0] + rf_conf_matrix[0][1])  # TN / (TN + FP)
rf_sensitivity = rf_recall  # Sensitivity is the same as Recall
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100  # MAPE

# Calculate the evaluation metrics for Decision Tree
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_conf_matrix = confusion_matrix(y_test, dt_pred)
dt_specificity = dt_conf_matrix[0][0] / (dt_conf_matrix[0][0] + dt_conf_matrix[0][1])  # TN / (TN + FP)
dt_sensitivity = dt_recall  # Sensitivity is the same as Recall
dt_mape = np.mean(np.abs((y_test - dt_pred) / y_test)) * 100  # MAPE

# Print results for Random Forest
print("Random Forest Evaluation Metrics:")
print(f"AUC: {rf_auc:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"Specificity: {rf_specificity:.4f}")
print(f"Sensitivity: {rf_sensitivity:.4f}")
print(f"MAPE: {rf_mape:.4f}%\n")

# Print results for Decision Tree
print("Decision Tree Evaluation Metrics:")
print(f"AUC: {dt_auc:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall: {dt_recall:.4f}")
print(f"Specificity: {dt_specificity:.4f}")
print(f"Sensitivity: {dt_sensitivity:.4f}")
print(f"MAPE: {dt_mape:.4f}%")

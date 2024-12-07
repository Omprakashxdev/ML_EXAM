# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:11:58 2024
5.	Implement and compare SVM, KNN and Logistic regression algorithm to
 classify the Android mobile purchase records data set for predicting
 both correct and wrong predictions.
@author: vasit
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
data = pd.read_csv('android_purchase_records.csv')

# Data preparation
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
svm_model = SVC(probability=True)
knn_model = KNeighborsClassifier(n_neighbors=5)
logreg_model = LogisticRegression()

# Train models
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
logreg_model.fit(X_train, y_train)

# Predictions
svm_preds = svm_model.predict(X_test)
knn_preds = knn_model.predict(X_test)
logreg_preds = logreg_model.predict(X_test)

# Evaluation
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("KNN Classification Report:")
print(classification_report(y_test, knn_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_preds))

print("Logistic Regression Classification Report:")
print(classification_report(y_test, logreg_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logreg_preds))

# ROC-AUC (Example for SVM)
svm_probs = svm_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, svm_probs)
print(f"SVM ROC-AUC: {roc_auc}")

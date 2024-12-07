# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:46:02 2024
3.	Write a program to implement the k-Nearest Neighbour algorithm to classify the iris to predict correct and wrong predictions. Use Python ML library classes for the prediction.
@author: vasit
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target (classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier (k=3)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of k-NN Classifier (k={k}): {accuracy:.2f}")

# Display correct and wrong predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results['Correct'] = results['Actual'] == results['Predicted']
print("\nPrediction Results:\n", results)

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#pip install scikit-learn

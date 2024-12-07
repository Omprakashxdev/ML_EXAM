# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:26:49 2024
6.	Write a python code to implement K-nearest neighbor for the given dataset. 
Assume that value of K=2. Calculate the accuracy and predict the Shirt size for the single customer using 
the below given sample dataset
@author: vasit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Height': [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170],
    'Weight': [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68],
    'T_Shirt_Size': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (Height and Weight) and Target (T_Shirt_Size)
X = df[['Height', 'Weight']]
y = df['T_Shirt_Size']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model with K=2
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict T-shirt size for a single customer
new_customer = [[161, 61]]  # Example customer with Height=161 and Weight=61
predicted_size = knn.predict(new_customer)
print(f"Predicted T-shirt size for the new customer: {predicted_size[0]}")

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:44:35 2024
9.	Write a python code for the below mentioned dataset to perform:
a.	To calculate missing values of every feature
b.	Clean and pre-process the dataset by removing the missing values
c.	Impute the missing values with Mean, Median, Mode
d.	Encode Categorical Data
e.	Find the accuracy using Linear Regression Model
f.	Represent the confusion matrix
g.	Represent in a graph


@author: vasit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Dataset
data = {
    'Country': ['India', 'Pakistan', 'Bhutan', 'Bangladesh', 'Nepal', 'Srilanka', 'Burma', 'China', 'Afganistan'],
    'Age': [49, 32, 35, 43, 45, 40, np.nan, 53, 55],
    'Salary': [62000, 38000, 44000, 51000, np.nan, 48000, 42000, 69000, 73000],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# a. Calculate missing values of every feature
missing_values = df.isnull().sum()
print("Missing Values in each column:")
print(missing_values)

# b. Remove rows with missing values
df_dropped = df.dropna()
print("\nData after removing rows with missing values:")
print(df_dropped)

# c. Impute missing values with Mean, Median, and Mode
df_imputed = df.copy()
df_imputed['Age'].fillna(df_imputed['Age'].mean(), inplace=True)
df_imputed['Salary'].fillna(df_imputed['Salary'].median(), inplace=True)
print("\nData after imputing missing values:")
print(df_imputed)

# d. Encode Categorical Data
label_encoders = {}
for column in ['Country', 'Purchased']:
    le = LabelEncoder()
    df_imputed[column] = le.fit_transform(df_imputed[column])
    label_encoders[column] = le
print("\nData after encoding categorical variables:")
print(df_imputed)

# Features (X) and Target (y)
X = df_imputed[['Country', 'Age', 'Salary']]
y = df_imputed['Purchased']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# e. Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_class = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# f. Represent the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("\nConfusion Matrix:")
print(conf_matrix)

# g. Represent in a graph
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoders['Purchased'].classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

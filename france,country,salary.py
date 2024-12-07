# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:51:09 2024
11.	Write the python code for the following dataset which is mentioned below,
a.	Import the libraries
b.	Import the data-set
c.	Check out the missing values
d.	See the Categorical Values
e.	Splitting the data-set into Training and Test Set
f.	Feature Scaling

@author: vasit
"""

# a. Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# b. Import the dataset
data = {
    'Country': ['France', 'Span', 'German', 'German', 'France', 'Span', 'German', 'Span', 'France', 'Span'],
    'Age': [44.00, 44.00, 54.00, 34.00, 44.00, 34.00, 23.00, 45.00, 23.00, 23.00],
    'Salary': [44000, 50000, 23000, np.nan, np.nan, 23667, 67000, 45000, 56000, 66000],
    'Purchased': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No']
}

# Create a DataFrame
df = pd.DataFrame(data)

# c. Check out the missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# d. See the Categorical Values
print("\nCategorical Columns in the dataset:")
categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

# e. Splitting the dataset into Training and Test Set
X = df.drop(columns=['Purchased'])  # Features
y = df['Purchased']  # Target variable

# Convert categorical values using Label Encoding
X = pd.get_dummies(X, drop_first=True)  # Convert categorical columns (Country) into numerical columns
y = y.map({'Yes': 1, 'No': 0})  # Map the 'Purchased' column to numerical values (1: Yes, 0: No)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# f. Feature Scaling (Standardization)
scaler = StandardScaler()

# Fit on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the scaled feature values
print("\nScaled Features for Training Set:")
print(X_train_scaled)

# The dataset has been prepared with feature scaling applied.

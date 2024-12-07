# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:29:53 2024

@author: vasit
"""
"""1.	Read the .csv file and calculate the missing value of each column 
and handle the missing values with imputation techniques."""


import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'your_file.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Display missing values for each column
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Define imputation strategies
# For numerical columns, we use mean imputation
numerical_imputer = SimpleImputer(strategy='mean')
numerical_columns = data.select_dtypes(include=['number']).columns

# For categorical columns, we use the most frequent value (mode)
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_columns = data.select_dtypes(exclude=['number']).columns

# Apply imputation
if not numerical_columns.empty:
    data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

if not categorical_columns.empty:
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Verify imputation
print("\nData after imputation:\n", data.head())

# Save the cleaned data (optional)
output_path = 'cleaned_data.csv'  # Replace with your desired output file path
data.to_csv(output_path, index=False)
print(f"\nCleaned data saved to {output_path}")




#pip install pandas scikit-learn

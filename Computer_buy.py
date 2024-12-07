# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:41:45 2024
8.	Write a python code to apply a classification algorithm to classify whether a person can buy a computer or not based on given test data :
Test Data
Age : Youth Income : Low  Student : No  Credit Rating : Fair  Buy Computer - ??


@author: vasit
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Dataset
data = {
    'ItemNo': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Middle', 'Senior', 'Youth', 'Youth', 'Senior', 
            'Youth', 'Middle', 'Middle', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'Credit': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Fair', 
               'Excellent', 'Excellent', 'Fair', 'Excellent'],
    'BuyComputer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Drop ItemNo (not a feature)
df = df.drop(columns=['ItemNo'])

# Encode categorical variables
label_encoders = {}
for column in ['Age', 'Income', 'Student', 'Credit', 'BuyComputer']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features (X) and target (y)
X = df.drop(columns=['BuyComputer'])
y = df['BuyComputer']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict for the new test data
test_data = {'Age': ['Youth'], 'Income': ['Low'], 'Student': ['No'], 'Credit': ['Fair']}
test_data_df = pd.DataFrame(test_data)

# Encode test data
for column in test_data_df.columns:
    test_data_df[column] = label_encoders[column].transform(test_data_df[column])

predicted_class = model.predict(test_data_df)
predicted_label = label_encoders['BuyComputer'].inverse_transform(predicted_class)
print(f"Predicted class for the test data (Buy Computer): {predicted_label[0]}")

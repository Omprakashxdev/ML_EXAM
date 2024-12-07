# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:37:42 2024
7.	Write a python code to implement NaiveBayes for the given dataset. Calculate the accuracy and predict the class for individual temperature factor?
	


@author: vasit
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Mild', 'Hot', 'Hot', 'Hot', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'High', 'High', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'], drop_first=True)

# Features (X) and target (y)
X = df_encoded.drop(columns=['Play Tennis'])
y = df['Play Tennis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict "Play Tennis" for an individual temperature factor
new_data = {'Outlook_Sunny': [0], 'Outlook_Overcast': [0], 'Temperature_Mild': [1], 'Humidity_Low': [0], 'Wind_Strong': [0]}
new_data_df = pd.DataFrame(new_data)

predicted_class = model.predict(new_data_df)
print(f"Predicted class for the individual: {predicted_class[0]}")

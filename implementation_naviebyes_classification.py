# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:34:24 2024

2.	Implement the naïve Bayesian classifier for a sample training data set stored in a .CSV file. Compute the accuracy of the classifier, considering few test data sets.
@author: vasit
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'your_file.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Check the data structure
print("Dataset:\n", data.head())

# Split the data into features and target
# Replace 'target_column' with the name of your target column
X = data.drop(columns=['target_column'])  # Features
y = data['target_column']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naïve Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Naïve Bayes Classifier: {accuracy:.2f}")

# Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save predictions (optional)
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")


#pip install pandas scikit-learn


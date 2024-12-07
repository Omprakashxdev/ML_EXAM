# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:49:56 2024
4.	Write a program to implement a Logistic Regression algorithm to
 classify the housing price data set to predict correct and wrong predictions.
 Use Python ML library classes for predicting the problem. 
@author: vasit
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = 'housing_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Display dataset structure
print("Dataset:\n", data.head())

# Define features (X) and target (y)
# Replace 'price_category' with the actual target column in your dataset
X = data.drop(columns=['price_category'])  # Features
y = data['price_category']  # Target (e.g., 0 for affordable, 1 for expensive)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of Logistic Regression Classifier: {accuracy:.2f}")

# Display correct and wrong predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results['Correct'] = results['Actual'] == results['Predicted']
print("\nPrediction Results:\n", results)

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#pip install pandas scikit-learn


"""
Ensure the target column is binary or categorical and encoded (e.g., 0 or 1). If not, use the following code for encoding:

python
Copy code
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['price_category'] = le.fit_transform(data['price_category'])
Run the script and let me know if you need help refining it for your dataset!
"""




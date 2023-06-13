from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score


import numpy as np

# Read the CSV file
dirty_data = pd.read_csv('loan-train.csv')

# Remove rows with missing values
data = dirty_data.dropna()

# Print the cleaned dataframe
# print(data)

X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

# Convert categorical variables to numerical using label encoding
encoder = LabelEncoder()
for feature in X.columns:
    if X[feature].dtype == 'object':
        X[feature] = encoder.fit_transform(X[feature])

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Predict the labels
y_pred = model.predict(X)
# print(y_pred)

# Calculate the accuracy
acc = accuracy_score(y, y_pred)
print(acc)

# testing it on another data
dirt_test_data = pd.read_csv('loan-test.csv')
test_data = dirt_test_data.dropna()

X_test = test_data.drop(columns=['Loan_ID', 'Loan_Status'])
y_test = test_data['Loan_Status']

# Apply the trained model to make predictions on the testing dataset
for feature in X_test.columns:
    if X_test[feature].dtype == 'object':
        X_test[feature] = encoder.fit_transform(X_test[feature])

predictions = model.predict(X_test)
# print(predictions)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label='Y')
recall = recall_score(y_test, predictions, pos_label='Y')
f1 = f1_score(y_test, predictions, pos_label='Y')


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
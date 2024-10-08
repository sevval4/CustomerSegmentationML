# 1. Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("Mall_Customers.csv")

# Convert Gender column to numerical
labelencoder = LabelEncoder()
data["Gender"] = labelencoder.fit_transform(data["Gender"])

# Define input features (excluding CustomerID)
X = data[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Create target variable (categorizing Spending Score)
data["Spending Category"] = pd.cut(
    data["Spending Score (1-100)"], bins=[0, 33, 66, 100], labels=[0, 1, 2]
)
Y = data["Spending Category"]  # Categorical target variable

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80%-20% ratio)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, train_size=0.8)

# 2. Neural Network (NN) Model
import tensorflow as tf
from tensorflow import keras

# Define the NN Model
model = tf.keras.models.Sequential()
model.add(
    keras.layers.Dense(512, input_shape=(4,), name="Hidden_layer_1", activation="relu")
)
model.add(keras.layers.Dense(512, name="Hidden_layer_2", activation="relu"))
model.add(keras.layers.Dense(128, name="Hidden_layer_3", activation="relu"))
model.add(
    keras.layers.Dense(3, name="Output_layer", activation="softmax")
)  # Softmax for 3 classes
model.compile(
    loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)  # Using sparse_categorical_crossentropy

# Train the model
history = model.fit(X_train, Y_train, epochs=30, batch_size=8, validation_split=0.1)

# 3. Other Machine Learning Algorithms
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# SVM Model
svm_model = SVC()
svm_model.fit(X_train, Y_train)
svm_predictions = svm_model.predict(X_test)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
dt_predictions = dt_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)
rf_predictions = rf_model.predict(X_test)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluate Results
print("SVM Results:\n", classification_report(Y_test, svm_predictions))
print("Decision Tree Results:\n", classification_report(Y_test, dt_predictions))
print("Random Forest Results:\n", classification_report(Y_test, rf_predictions))
print("Logistic Regression Results:\n", classification_report(Y_test, lr_predictions))

from sklearn.model_selection import GridSearchCV

# Hyperparameter optimization for SVM
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, Y_train)
print("Best parameters:", grid_search.best_params_)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Predictions using NN
Y_pred_nn = model.predict(X_test)
Y_pred_nn_classes = np.argmax(Y_pred_nn, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_nn_classes)

# Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("NN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# NN evaluation report
print(classification_report(Y_test, Y_pred_nn_classes))
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy Improvements with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Model predictions with test data
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Create confusion matrix
cm = confusion_matrix(Y_test, y_pred_classes)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix for Test Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Count of approved and rejected applications
unique, counts = np.unique(y_pred_classes, return_counts=True)
loan_status_counts = dict(zip(unique, counts))

# Set class names correctly
labels = [
    "Approved" if label == 1 else "Rejected" for label in loan_status_counts.keys()
]

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    loan_status_counts.values(),
    labels=labels,
    autopct="%1.1f%%",
    colors=["tomato", "orange"],
)
plt.title("Model Predictions (Approved vs Rejected)")
plt.show()

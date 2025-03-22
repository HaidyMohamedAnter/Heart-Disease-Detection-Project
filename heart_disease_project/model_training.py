import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Step 1: Load the Cleaned Data
data = pd.read_csv("cleaned_data.csv")
print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# Step 2: Split Data (80% Training, 20% Testing)
X = data.drop(columns=["target"])  # Features
y = data["target"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data Split Completed! Train Size:", X_train.shape, "Test Size:", X_test.shape)

# Save the split data to CSV files
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Split Data Saved to CSV Files!")

# Step 3: Train the Decision Tree Model (Initial Model)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Initial Decision Tree Model Trained!")

# Step 4: Hyperparameter Tuning (Optimize Depth and Min Samples per Split)
param_grid = {
    "max_depth": [3, 5, 10, None],  # Test different depths, including unlimited (None)
    "min_samples_split": [2, 5, 10],  # Test different minimum samples for splitting
    "min_samples_leaf": [1, 2, 4]  # Added min_samples_leaf for better control over leaf nodes
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),  # Base model
    param_grid=param_grid,  # Hyperparameter grid
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",  # Metric to optimize
    n_jobs=-1  # Use all available CPU cores for faster computation
)

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Retrieve the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Step 5: Evaluate the Model on Test Data
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Print evaluation results
print("Model Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the Model
joblib.dump(best_model, "decision_tree_model.pkl")
print("Model Saved Successfully!")

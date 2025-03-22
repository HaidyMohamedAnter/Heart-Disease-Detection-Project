import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
def load_dataset(filename):
    """Loads the heart disease dataset."""
    return pd.read_csv(filename)

# Handle Missing Values
def handle_missing_values(df):
    """Fills missing values with median or drops rows if necessary."""
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

# Normalize Data
def normalize_data(df, numerical_features):
    """Scales numerical features using MinMaxScaler."""
    existing_features = [col for col in numerical_features if col in df.columns]
    if not existing_features:
        raise ValueError("No valid numerical features found!")
    
    scaler = MinMaxScaler()
    df[existing_features] = scaler.fit_transform(df[existing_features])
    return df

# Encode Categorical Variables
def encode_categorical(df, categorical_features):
    """Encodes categorical features using One-Hot Encoding."""
    df = pd.get_dummies(df, columns=[col for col in categorical_features if col in df.columns], drop_first=True)
    return df

# Feature Selection
def feature_selection(df):
    """Performs correlation analysis to identify important features."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    return df

# Save Cleaned Data
def save_cleaned_data(df, filename="cleaned_data.csv"):
    """Saves the processed dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Cleaned data saved as {filename}")

# Main Function
def main():
    filename = "heart.csv"
    df = load_dataset(filename)
    
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    df = handle_missing_values(df)
    df = normalize_data(df, numerical_features)
    df = encode_categorical(df, categorical_features)
    df = feature_selection(df)
    save_cleaned_data(df)

if __name__ == "__main__":
    main()
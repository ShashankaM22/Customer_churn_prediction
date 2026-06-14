"""
Data Preprocessing Pipeline for Telco Customer Churn Prediction.

This module provides functions to load, clean, split, and preprocess the 
Telco Customer Churn dataset. It prepares features for model training and inference.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV dataset.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning operations:
    1. Drop customerID (not useful for prediction).
    2. Convert TotalCharges to numeric, handling empty/whitespace values.
    3. Fill missing values if any.
    4. Convert target variable 'Churn' to binary (Yes=1, No=0).
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    
    # Drop identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # TotalCharges contains empty spaces, convert to NaN and then fill/impute
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
        # Impute missing TotalCharges with Median (or MonthlyCharges * tenure if tenure > 0)
        median_total_charges = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
        
    # Convert target variable Churn to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def get_feature_lists(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Identify categorical and numerical features.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        target_col (str): The label column to predict.
        
    Returns:
        tuple: (list of numerical columns, list of categorical columns)
    """
    features = [col for col in df.columns if col != target_col]
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # If senior citizen is encoded as 0/1, it can be treated as categorical or numeric
    # Let's keep it categorical for encoder
    categorical_cols = [col for col in features if col not in numerical_cols]
    
    return numerical_cols, categorical_cols

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    target_col: str = 'Churn', save_preprocessor_path: str = None):
    """
    Preprocess features: One-Hot encode categorical features and scale numerical features.
    Saves the preprocessor pipeline for future prediction.
    
    Args:
        train_df (pd.DataFrame): Training set.
        test_df (pd.DataFrame): Test set.
        target_col (str): Name of target column.
        save_preprocessor_path (str): Filepath to save the preprocessor (e.g., preprocessor.pkl).
        
    Returns:
        tuple: (X_train_processed, y_train, X_test_processed, y_test, preprocessor)
    """
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    numerical_cols, categorical_cols = get_feature_lists(train_df, target_col)
    
    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Fit and transform training data, transform test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor for inference
    if save_preprocessor_path:
        os.makedirs(os.path.dirname(save_preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, save_preprocessor_path)
        print(f"Preprocessor saved to {save_preprocessor_path}")
        
    # Get column names after transformation for feature importance analysis
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_features = numerical_cols + encoded_cat_cols
    
    return X_train_processed, y_train, X_test_processed, y_test, preprocessor, all_features

if __name__ == '__main__':
    print("Prepossessing module defined. Run notebooks or train.py to preprocess data.")

"""
Training Pipeline for Customer Churn Prediction.

This script loads the Telco Customer Churn dataset, cleans it, splits it,
preprocesses it, trains both Logistic Regression and Random Forest models,
evaluates both using multiple classification metrics, prints a comparison table,
and saves the best-performing model based on ROC-AUC.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib

# Import preprocessing pipeline
from data_preprocessing import load_data, clean_data, preprocess_data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using multiple classification metrics.
    
    Args:
        model: Trained model object.
        X_test: Preprocessed test features.
        y_test: Test labels.
        
    Returns:
        dict: Dictionary of metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def run_training_pipeline(data_path: str, model_save_dir: str = 'models'):
    """
    Run the full training, evaluation, comparison, and saving pipeline.
    
    Args:
        data_path (str): Path to raw CSV file.
        model_save_dir (str): Directory where trained artifacts will be saved.
    """
    print("=== STEP 1: LOADING DATA ===")
    df = load_data(data_path)
    print(f"Data successfully loaded. Shape: {df.shape}")
    
    print("\n=== STEP 2: CLEANING DATA ===")
    df_cleaned = clean_data(df)
    print("Data cleaned. Missing values in TotalCharges handled.")
    
    print("\n=== STEP 3: SPLITTING DATA ===")
    # Stratified split to preserve class proportions in train/test sets
    train_df, test_df = train_test_split(
        df_cleaned, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_cleaned['Churn']
    )
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print("\n=== STEP 4: PREPROCESSING DATA ===")
    preprocessor_path = os.path.join(model_save_dir, 'preprocessor.pkl')
    X_train, y_train, X_test, y_test, preprocessor, feature_names = preprocess_data(
        train_df, test_df, target_col='Churn', save_preprocessor_path=preprocessor_path
    )
    print(f"Features preprocessed. Number of features: {X_train.shape[1]}")
    
    # Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    }
    
    results = {}
    
    print("\n=== STEP 5: TRAINING & EVALUATING MODELS ===")
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Save individual model
        filename = model_name.lower().replace(" ", "_") + "_model.pkl"
        model_path = os.path.join(model_save_dir, filename)
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to {model_path}")
        
        # Evaluate model
        results[model_name] = evaluate_model(model, X_test, y_test)
        
    print("\n=== STEP 6: MODEL PERFORMANCE COMPARISON ===")
    # Construct summary table
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1-Score': f"{metrics['F1-Score']:.4f}",
            'ROC-AUC': f"{metrics['ROC-AUC']:.4f}"
        })
    
    comparison_df = pd.DataFrame(summary_data)
    print("\n" + comparison_df.to_string(index=False) + "\n")
    
    # Save comparison report to CSV
    comparison_path = os.path.join(model_save_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison summary saved to {comparison_path}")
    
    # Display Confusion Matrices
    for model_name, metrics in results.items():
        cm = metrics['Confusion Matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix for {model_name}:")
        print(f"   Predicted No   Predicted Yes")
        print(f"Actual No:   {tn:<8}     {fp:<8}")
        print(f"Actual Yes:  {fn:<8}     {tp:<8}")
        
    # Save the Best Model based on ROC-AUC
    best_model_name = max(results, key=lambda k: results[k]['ROC-AUC'])
    best_model = models[best_model_name]
    best_model_path = os.path.join(model_save_dir, 'best_model.pkl')
    
    joblib.dump(best_model, best_model_path)
    print(f"\n=== STEP 7: BEST MODEL SAVED ===")
    print(f"Best model: {best_model_name} (ROC-AUC: {results[best_model_name]['ROC-AUC']:.4f})")
    print(f"Best model saved to {best_model_path}")
    
    # Write a quick text log of the comparison in the models directory
    with open(os.path.join(model_save_dir, 'results_summary.txt'), 'w') as f:
        f.write("Customer Churn Prediction - Model Comparison\n")
        f.write("============================================\n\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        f.write(f"Best Model: {best_model_name} (Saved as best_model.pkl)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Customer Churn Prediction Models")
    parser.add_argument('--data_path', type=str, default='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv', 
                        help='Path to raw dataset CSV')
    parser.add_argument('--model_save_dir', type=str, default='models', 
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    try:
        run_training_pipeline(args.data_path, args.model_save_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please verify that the raw Telco customer dataset CSV exists.")

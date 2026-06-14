"""
Inference script for Customer Churn Prediction.

This module loads the trained model and preprocessor to perform churn prediction
on new customer records. It can be run from the command line or imported as a utility.
"""

import os
import argparse
import pandas as pd
import joblib

def predict_churn(customer_data: dict, model_path: str = 'models/best_model.pkl', 
                  preprocessor_path: str = 'models/preprocessor.pkl') -> dict:
    """
    Predict churn for a single customer record.
    
    Args:
        customer_data (dict): A dictionary of customer attributes matching the dataset structure.
        model_path (str): Path to the serialized trained model pickle file.
        preprocessor_path (str): Path to the serialized ColumnTransformer preprocessor.
        
    Returns:
        dict: Churn prediction (0/1) and the associated probability of churning.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run train.py first.")
        
    # Load artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Convert input to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Preprocess incoming feature set (skip target cleaning as we don't have a label here)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Standardize column types if necessary (e.g., SeniorCitizen, TotalCharges)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Simple fallback for missing total charges
        if df['TotalCharges'].isnull().any():
            df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
            
    # Transform features
    X_processed = preprocessor.transform(df)
    
    # Make prediction
    prediction = int(model.predict(X_processed)[0])
    probability = float(model.predict_proba(X_processed)[0][1])
    
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Customer Churn")
    parser.add_argument('--model_path', type=str, default='models/best_model.pkl',
                        help='Path to the trained model (.pkl)')
    parser.add_argument('--preprocessor_path', type=str, default='models/preprocessor.pkl',
                        help='Path to the preprocessor (.pkl)')
    
    args = parser.parse_args()
    
    # Sample customer data representing a customer who is highly likely to churn:
    # - Month-to-month contract
    # - Short tenure (1 month)
    # - High monthly charges
    # - Fiber optic internet, no additional security
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.05,
        "TotalCharges": 70.05
    }
    
    print("Inference configuration loaded.")
    print("\nRunning test prediction on sample customer:")
    for k, v in sample_customer.items():
        print(f"  {k}: {v}")
        
    try:
        result = predict_churn(sample_customer, args.model_path, args.preprocessor_path)
        print("\n=== Prediction Result ===")
        print(f"Churn Prediction: {result['churn_prediction']}")
        print(f"Churn Probability: {result['churn_probability'] * 100:.2f}%")
    except FileNotFoundError as e:
        print(f"\nCould not run test prediction: {e}")
        print("Tip: Train the model first to generate the models directory and pickle files.")

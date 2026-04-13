import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import os

# 1. The Asset Loader
def load_resources():
    model_path = 'modeling/xgb_model.json'
    scaler_path = 'modeling/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing files! Check {model_path} or {scaler_path}")

    model = xgb.Booster()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, explainer

# 2. The Preprocessing Hook (The "Feature Lock" Fix)
def preprocess_data(data_row, scaler):
    # We explicitly define the 17 features used during training.
    # We exclude 'cik', 'ddate', and 'target_crash'.
    features = [
        'Assets', 'Revenues', 'NetIncomeLoss', 'current_ratio', 'quick_ratio', 
        'cash_ratio', 'roa', 'profit_margin', 'operating_margin', 'roe', 
        'debt_to_assets', 'debt_to_equity', 'asset_turnover', 'interest_coverage', 
        'retained_earnings_ratio', 'revenue_growth_rate', 'persistent_distress_flag'
    ]
    
    # Select only the 17 features in this exact order
    numeric_data = data_row[features]
    
    # Scale the data
    scaled_data = scaler.transform(numeric_data)
    
    # Create the DMatrix for XGBoost
    return scaled_data, xgb.DMatrix(scaled_data, feature_names=features)

# 3. The Prediction Logic
def get_prediction(dmatrix, model):
    probability = model.predict(dmatrix)[0]
    status = "High Risk" if probability >= 0.5 else "Stable"
    return status, round(float(probability), 4)

# 4. The SHAP Logic
def get_explanation(scaled_data, explainer):
    shap_values = explainer.shap_values(scaled_data)
    return shap_values

# 5. The Safety Test Block
if __name__ == "__main__":
    try:
        print("Starting Final Bulletproof Test...")
        m, s, e = load_resources()
        
        data_path = '04_master_ml/lstm_ready_data.parquet'
        test_df = pd.read_parquet(data_path).iloc[0:1]
        
        print(f"Testing Company: {test_df.iloc[0].get('Entity Name', 'Unknown')}")
        
        # Process, Predict, and Explain
        scaled_arr, dmat = preprocess_data(test_df, s)
        res, prob = get_prediction(dmat, m)
        s_vals = get_explanation(scaled_arr, e)
        
        print("-" * 30)
        print(f"RESULT: {res}")
        print(f"PROBABILITY: {prob}")
        print(f"SHAP: {len(s_vals[0])} features analyzed.")
        print("-" * 30)
        print("Everything is working perfectly!")
        
    except Exception as err:
        print(f"Critical Error found: {err}")
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Card Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float

def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'output', 'model.joblib')
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at {model_path}. Please train the model first."
            )
            
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['scaler'], model_data['feature_cols']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """Make a prediction for a single transaction."""
    try:
        model, scaler, feature_cols = load_model_and_scaler()
        
        # Create a DataFrame with the transaction data
        df = pd.DataFrame([transaction.dict()])
        
        # Calculate derived features
        df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['amount_percent_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        
        # Encode transaction type
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['type_encoded'] = le.fit_transform(df['type'])
        
        # Prepare features
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    try:
        with open("src/static/index.html", "r") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading index.html: {str(e)}") 
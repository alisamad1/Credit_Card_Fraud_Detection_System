import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the credit card transaction data."""
    logger.info(f"Loading data from {file_path}...")
    # Use chunksize to load data in chunks
    chunk_size = 100000
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    return data

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    logger.info("Handling missing values...")
    numeric_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df['nameDest'] = df['nameDest'].fillna('UNKNOWN')
    fraud_cols = ['isFraud', 'isFlaggedFraud']
    df[fraud_cols] = df[fraud_cols].fillna(0)
    return df

def engineer_features(df):
    """Create new features for better fraud detection."""
    logger.info("Engineering features...")
    # Calculate balance differences
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Calculate transaction amount as percentage of original balance
    df['amount_percent_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    # Create interaction features
    df['amount_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    # Encode transaction type
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    return df

def prepare_features(df):
    """Prepare features for model training."""
    logger.info("Preparing features for training...")
    feature_cols = [
        'step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig',
        'balance_diff_dest', 'amount_percent_orig', 'amount_balance_ratio'
    ]
    
    X = df[feature_cols]
    y = df['isFraud']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols

def preprocess_data(file_path):
    """Complete preprocessing pipeline."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = engineer_features(df)
    X, y, scaler, feature_cols = prepare_features(df)
    return X, y, scaler, feature_cols 
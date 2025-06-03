import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def train_model(X, y):
    """Train a Random Forest model with SMOTE for handling class imbalance."""
    logger.info("Starting model training...")
    
    # Split the data
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to handle class imbalance
    logger.info("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42, sampling_strategy=0.1)  # Reduced sampling ratio
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train the model with optimized parameters
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=8,      # Reduced from 10
        min_samples_split=100,  # Added to reduce complexity
        min_samples_leaf=50,    # Added to reduce complexity
        n_jobs=-1,        # Use all available CPU cores
        random_state=42,
        class_weight='balanced',
        verbose=1         # Show progress
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    logger.info("Model training completed!")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logger.info("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'roc_auc_score': roc_auc
    }

def save_model(model, scaler, feature_cols, model_path='model.joblib'):
    """Save the trained model and scaler."""
    logger.info(f"Saving model to {model_path}...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, model_path)
    logger.info("Model saved successfully!")

def load_model(model_path='model.joblib'):
    """Load the trained model and scaler."""
    return joblib.load(model_path) 
from data_preprocessing import preprocess_data
from model import train_model, evaluate_model, save_model
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to train and evaluate the model."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Preprocess the data
        logger.info("Preprocessing data...")
        X, y, scaler, feature_cols = preprocess_data('card.csv.csv')
        
        # Train the model
        logger.info("Training model...")
        model, X_test, y_test = train_model(X, y)
        
        # Evaluate the model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save the model
        logger.info("Saving model...")
        save_model(model, scaler, feature_cols, 'output/model.joblib')
        
        # Log metrics
        logger.info("\nModel Evaluation Metrics:")
        logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
        logger.info(f"\nROC AUC Score: {metrics['roc_auc_score']:.4f}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
# Credit Card Fraud Detection System

This project implements a machine learning-based system for detecting fraudulent credit card transactions. The system uses a Random Forest classifier with SMOTE for handling class imbalance and includes a FastAPI-based API for making predictions.

## Dataset

The dataset used in this project is not included in the repository due to its large size. You can download it from:
- [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the downloaded dataset in the root directory of the project.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── api.py
│   └── train.py
├── output/
│   └── model.joblib
└── card.csv  # Download from the link above
```

## Features

- Data preprocessing and feature engineering
- SMOTE for handling class imbalance
- Random Forest classifier for fraud detection
- Model evaluation with multiple metrics
- FastAPI-based REST API for predictions
- Comprehensive logging
- Model persistence

## Installation

1. Clone the repository
2. Download the dataset from the link above and place it in the root directory
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model:

```bash
python src/train.py
```

This will:
- Preprocess the data
- Train the model
- Evaluate the model
- Save the model to `output/model.joblib`

### Running the API

To start the API server:

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://localhost:8000`

### Making Predictions

Send a POST request to `/predict` with transaction data:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "step": 1,
           "type": "PAYMENT",
           "amount": 9839.64,
           "oldbalanceOrg": 170136.0,
           "newbalanceOrig": 160296.36,
           "oldbalanceDest": 0.0,
           "newbalanceDest": 0.0
         }'
```

## API Documentation

Once the API is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Model Performance

The model's performance is evaluated using:
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC AUC Score

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
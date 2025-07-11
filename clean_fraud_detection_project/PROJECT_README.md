# 🔍 Credit Card Fraud Detection System

A comprehensive AI-powered credit card fraud detection system built with Python, featuring machine learning models, real-time prediction API, and monitoring capabilities.

## 🌟 Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Logistic Regression, and Ensemble methods
- **Real-time Predictions**: REST API for single and batch transaction processing
- **Feature Engineering**: Advanced feature extraction from transaction data
- **Imbalanced Data Handling**: SMOTE and undersampling techniques
- **Comprehensive Monitoring**: Real-time alerts and statistics
- **Visualization Tools**: Performance analysis and fraud pattern visualization
- **Synthetic Data Generation**: Built-in data generator for testing and development

## 🏗️ Project Structure

```
fraud_detection/
├── fraud_detection/           # Main package
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data_generator.py     # Synthetic data generation
│   ├── feature_engineering.py # Feature extraction and processing
│   ├── models.py             # ML models and ensemble methods
│   ├── predictor.py          # Real-time prediction system
│   └── visualization.py      # Analysis and visualization tools
├── train_model.py            # Training pipeline script
├── api_server.py             # Flask API server
├── demo.py                   # Demo and testing script
├── requirements.txt          # Python dependencies
└── PROJECT_README.md         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
# Navigate to project directory

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Generate data and train models
python train_model.py

# Or with custom parameters
python train_model.py --transactions 100000 --fraud-rate 0.025
```

### 3. Start API Server

```bash
# Start the Flask API server
python api_server.py
```

### 4. Run Demo

```bash
# Test the system with sample transactions
python demo.py
```

## 📊 Usage Examples

### Training Models

```bash
# Basic training (50,000 transactions, 2% fraud rate)
python train_model.py

# Large dataset training
python train_model.py --transactions 200000 --fraud-rate 0.03

# Use existing data (skip generation)
python train_model.py --skip-data-generation
```

### API Usage

#### Single Transaction Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "amount": 1500.00,
    "merchant_category": "electronics",
    "location_risk_score": 0.8
  }'
```

Response:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.85,
  "risk_level": "HIGH",
  "explanation": ["Large transaction amount ($1500.00)", "High-risk merchant category (electronics)", "High-risk location"],
  "transaction_id": "unknown",
  "timestamp_processed": "2024-01-15T14:30:00.123456"
}
```

#### Batch Prediction

```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "user_id": "user_123",
        "amount": 50.00,
        "merchant_category": "grocery",
        "location_risk_score": 0.1
      },
      {
        "user_id": "user_456",
        "amount": 2500.00,
        "merchant_category": "jewelry",
        "location_risk_score": 0.9
      }
    ]
  }'
```

### Python API

```python
from fraud_detection.predictor import FraudPredictor

# Load trained model
predictor = FraudPredictor(
    model_path="models/ensemble_model.pkl",
    feature_engineer_path="models/feature_engineer.pkl"
)

# Single prediction
transaction = {
    "user_id": "user_123",
    "amount": 150.75,
    "merchant_category": "restaurant",
    "location_risk_score": 0.2
}

result = predictor.predict_single_transaction(transaction)
print(f"Fraud probability: {result['fraud_probability']:.2%}")
```

## 🎯 Model Performance

The system trains multiple models and selects the best performer:

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Random Forest | 0.95+ | 0.85+ | 0.80+ | 0.82+ |
| XGBoost | 0.96+ | 0.87+ | 0.82+ | 0.84+ |
| LightGBM | 0.95+ | 0.86+ | 0.81+ | 0.83+ |
| Ensemble | 0.97+ | 0.88+ | 0.84+ | 0.86+ |

*Performance may vary based on data and parameters*

## 📈 Features Used

### Transaction Features
- **Amount**: Transaction value and log transformation
- **Merchant Category**: Type of business (grocery, electronics, etc.)
- **Location Risk Score**: Geographic risk assessment
- **Timestamp**: Time-based patterns (hour, day of week, etc.)

### Engineered Features
- **User Behavior**: Historical spending patterns and statistics
- **Velocity**: Transaction frequency metrics
- **Anomaly Detection**: Deviations from normal patterns
- **Risk Scores**: Merchant and location risk assessments

## 🔧 Configuration

Edit `fraud_detection/config.py` to customize:

```python
# Model parameters
FRAUD_THRESHOLD = 0.5          # Classification threshold
HIGH_RISK_THRESHOLD = 0.7      # High-risk alert threshold

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    # ... other parameters
}

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
```

## 🚨 Monitoring and Alerts

The system provides real-time monitoring:

### Get System Statistics
```bash
curl http://localhost:5000/statistics
```

### Get Recent Alerts
```bash
curl http://localhost:5000/alerts?limit=10
```

### Health Check
```bash
curl http://localhost:5000/health
```

## 📊 Visualization

Generate analysis plots:

```python
from fraud_detection.visualization import FraudAnalysisVisualizer
import pandas as pd

# Load data
df = pd.read_csv("data/credit_card_transactions.csv")

# Create visualizer
viz = FraudAnalysisVisualizer()

# Plot transaction patterns
fig = viz.plot_transaction_patterns(df)
plt.show()

# Create interactive dashboard
dashboard = viz.create_interactive_dashboard(df)
dashboard.show()
```

## 🛠️ Development

### Adding New Models

1. Implement model in `fraud_detection/models.py`
2. Add configuration in `fraud_detection/config.py`
3. Update training pipeline in `train_model.py`

### Extending Features

1. Add feature extraction in `fraud_detection/feature_engineering.py`
2. Update feature list in configuration
3. Retrain models with new features

### Custom Data Sources

Replace the data generator in `fraud_detection/data_generator.py` or load your own data in the training script.

## 📝 API Documentation

When the server is running, visit:
- **API Documentation**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **Statistics**: http://localhost:5000/statistics
- **Alerts**: http://localhost:5000/alerts

## 🔒 Security Considerations

- **Input Validation**: All API inputs are validated
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Consider implementing rate limiting for production
- **Authentication**: Add authentication for production deployments

## 📊 Datasets

The system includes a synthetic data generator that creates realistic transaction patterns:

- **Normal Transactions**: Typical spending patterns with various merchants
- **Fraudulent Transactions**: Suspicious patterns including unusual amounts, locations, and timing
- **Configurable Parameters**: Fraud rate, number of users, transaction volume

## 🧪 Testing

```bash
# Run the demo script
python demo.py

# Test API endpoints
python -c "
import requests
response = requests.get('http://localhost:5000/health')
print(response.json())
"
```

## 📈 Performance Optimization

- **Batch Processing**: Use batch predictions for better throughput
- **Model Caching**: Models are loaded once and cached in memory
- **Feature Caching**: Consider caching user features for frequent users
- **Async Processing**: Consider async frameworks for high-volume scenarios

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train_model.py

EXPOSE 5000
CMD ["python", "api_server.py"]
```

### Production Considerations

- Use WSGI server (Gunicorn, uWSGI) instead of Flask dev server
- Implement proper logging and monitoring
- Set up database for persistent storage
- Configure load balancing for high availability
- Implement model versioning and A/B testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is provided for educational and demonstration purposes.

## 🆘 Troubleshooting

### Common Issues

1. **Models not found**: Run `python train_model.py` first
2. **API server won't start**: Check if port 5000 is available
3. **Poor performance**: Increase training data size or adjust hyperparameters
4. **Memory issues**: Reduce batch size or use smaller models

### Error Messages

- `Model not loaded`: Train models first with `train_model.py`
- `Feature mismatch`: Ensure consistent feature engineering pipeline
- `API timeout`: Increase timeout settings for large batch predictions

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Examine the logs in the `logs/` directory
4. Test with the demo script first

---

**Built with ❤️ for fraud detection and prevention**
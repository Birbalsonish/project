"""
Configuration settings for the fraud detection system.
"""

import os

# Data settings
DATA_PATH = "data/"
MODELS_PATH = "models/"
LOGS_PATH = "logs/"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Feature engineering
FEATURE_COLUMNS = [
    'amount', 'hour', 'day_of_week', 'is_weekend',
    'amount_mean_last_7_days', 'amount_std_last_7_days',
    'transaction_count_last_hour', 'transaction_count_last_day',
    'amount_normalized', 'merchant_category', 'location_risk_score'
]

# Thresholds
FRAUD_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG = True
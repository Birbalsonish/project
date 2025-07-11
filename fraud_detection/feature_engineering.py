"""
Feature engineering for credit card fraud detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer."""
        self.scalers = {}
        self.label_encoders = {}
        self.is_fitted = False
    
    def extract_time_features(self, df):
        """Extract time-based features from timestamp."""
        df = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Time-based risk factors
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def extract_user_features(self, df):
        """Extract user-based features using historical data."""
        df = df.copy()
        df = df.sort_values(['user_id', 'timestamp'])
        
        # User spending patterns (simpler approach)
        user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
        user_stats.columns = ['user_id', 'user_amount_mean', 'user_amount_std', 'user_transaction_count']
        user_stats['user_amount_std'] = user_stats['user_amount_std'].fillna(0)
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Simple rolling features (without time-based windows due to complexity)
        df['amount_mean_last_7_days'] = df.groupby('user_id')['amount'].rolling(window=7, min_periods=1).mean().reset_index(drop=True)
        df['amount_std_last_7_days'] = df.groupby('user_id')['amount'].rolling(window=7, min_periods=1).std().reset_index(drop=True).fillna(0)
        
        # Transaction frequency features (simplified)
        df['transaction_count_last_hour'] = df.groupby('user_id').cumcount() + 1
        df['transaction_count_last_day'] = df.groupby('user_id').cumcount() + 1
        
        return df
    
    def extract_amount_features(self, df):
        """Extract amount-based features."""
        df = df.copy()
        
        # Log transformation to handle skewness
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount normalized by user's historical mean
        df['amount_normalized'] = df['amount'] / (df['user_amount_mean'] + 1e-8)
        
        # Deviation from user's normal spending
        df['amount_deviation'] = np.abs(df['amount'] - df['user_amount_mean']) / (df['user_amount_std'] + 1e-8)
        
        # Amount percentiles
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=[0, 10, 50, 200, 1000, float('inf')], 
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        ).astype(str)
        
        return df
    
    def extract_merchant_features(self, df):
        """Extract merchant-based features."""
        df = df.copy()
        
        # Merchant category risk scores (based on domain knowledge)
        merchant_risk_map = {
            'grocery': 0.1, 'pharmacy': 0.1, 'gas_station': 0.2,
            'restaurant': 0.2, 'department_store': 0.3, 'hotel': 0.3,
            'car_rental': 0.4, 'airline': 0.3, 'entertainment': 0.5,
            'online_retail': 0.6, 'electronics': 0.7, 'jewelry': 0.8
        }
        
        df['merchant_risk_score'] = df['merchant_category'].map(merchant_risk_map).fillna(0.5)
        
        # Merchant category frequency for user
        merchant_user_counts = df.groupby(['user_id', 'merchant_category']).size().reset_index(name='user_merchant_count')
        df = df.merge(merchant_user_counts, on=['user_id', 'merchant_category'], how='left')
        df['user_merchant_count'] = df['user_merchant_count'].fillna(0)
        
        return df
    
    def extract_sequence_features(self, df):
        """Extract sequential pattern features."""
        df = df.copy()
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Time since last transaction (simplified)
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Amount change from previous transaction
        df['amount_change'] = df.groupby('user_id')['amount'].diff().fillna(0)
        
        # Velocity features (simplified)
        df['velocity_1h'] = 1.0  # Simplified
        df['velocity_24h'] = 1.0 / 24.0  # Simplified
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features."""
        df = df.copy()
        
        categorical_columns = ['merchant_category', 'amount_category']
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in unique_values else 'unknown')
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    else:
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def scale_numerical_features(self, df, fit=True):
        """Scale numerical features."""
        df = df.copy()
        
        numerical_columns = [
            'amount', 'amount_log', 'amount_normalized', 'amount_deviation',
            'amount_mean_last_7_days', 'amount_std_last_7_days',
            'time_since_last_transaction', 'velocity_1h', 'velocity_24h'
        ]
        
        for col in numerical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
                else:
                    if col in self.scalers:
                        df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
                    else:
                        df[f'{col}_scaled'] = 0
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform the dataset."""
        print("Engineering features...")
        
        # Extract all features
        df = self.extract_time_features(df)
        df = self.extract_user_features(df)
        df = self.extract_amount_features(df)
        df = self.extract_merchant_features(df)
        df = self.extract_sequence_features(df)
        
        # Encode and scale features
        df = self.encode_categorical_features(df, fit=True)
        df = self.scale_numerical_features(df, fit=True)
        
        self.is_fitted = True
        
        print(f"Feature engineering completed. Dataset shape: {df.shape}")
        return df
    
    def transform(self, df):
        """Transform new data using fitted encoders and scalers."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform.")
        
        # Extract all features
        df = self.extract_time_features(df)
        df = self.extract_user_features(df)
        df = self.extract_amount_features(df)
        df = self.extract_merchant_features(df)
        df = self.extract_sequence_features(df)
        
        # Encode and scale features
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_numerical_features(df, fit=False)
        
        return df
    
    def get_feature_columns(self):
        """Get the list of engineered feature columns for modeling."""
        return [
            'amount_scaled', 'amount_log_scaled', 'amount_normalized_scaled',
            'amount_deviation_scaled', 'amount_percentile', 'hour', 'day_of_week',
            'is_weekend', 'is_night', 'is_business_hours',
            'amount_mean_last_7_days_scaled', 'amount_std_last_7_days_scaled',
            'transaction_count_last_hour', 'transaction_count_last_day',
            'merchant_category_encoded', 'merchant_risk_score', 'location_risk_score',
            'time_since_last_transaction_scaled', 'velocity_1h_scaled', 'velocity_24h_scaled',
            'amount_category_encoded'
        ]
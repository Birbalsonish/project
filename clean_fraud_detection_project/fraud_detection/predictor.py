"""
Real-time fraud prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from .models import FraudDetectionModel, EnsembleModel
from .feature_engineering import FeatureEngineer
from .config import FRAUD_THRESHOLD, HIGH_RISK_THRESHOLD

class FraudPredictor:
    """Real-time fraud detection predictor."""
    
    def __init__(self, model_path=None, feature_engineer_path=None):
        """
        Initialize the fraud predictor.
        
        Args:
            model_path: Path to saved model
            feature_engineer_path: Path to saved feature engineer
        """
        self.model = None
        self.feature_engineer = None
        self.is_ready = False
        
        if model_path and feature_engineer_path:
            self.load_model(model_path, feature_engineer_path)
    
    def load_model(self, model_path, feature_engineer_path):
        """Load trained model and feature engineer."""
        try:
            # Load model (could be single model or ensemble)
            if 'ensemble' in model_path.lower():
                self.model = EnsembleModel.load_ensemble(model_path)
            else:
                self.model = FraudDetectionModel.load_model(model_path)
            
            # Load feature engineer
            import joblib
            self.feature_engineer = joblib.load(feature_engineer_path)
            
            self.is_ready = True
            print("Model and feature engineer loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_ready = False
    
    def predict_single_transaction(self, transaction_data):
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction_data: Dict with transaction details
        
        Returns:
            dict: Prediction results
        """
        if not self.is_ready:
            raise ValueError("Model not loaded. Call load_model first.")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Ensure required columns exist with defaults
            required_columns = ['user_id', 'timestamp', 'amount', 'merchant_category', 'location_risk_score']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'timestamp':
                        df[col] = datetime.now()
                    elif col == 'location_risk_score':
                        df[col] = 0.3  # Default medium risk
                    else:
                        raise ValueError(f"Missing required field: {col}")
            
            # Engineer features
            df_features = self.feature_engineer.transform(df)
            
            # Get feature columns for model
            feature_columns = self.feature_engineer.get_feature_columns()
            
            # Select only available features
            available_features = [col for col in feature_columns if col in df_features.columns]
            
            if not available_features:
                raise ValueError("No valid features found for prediction.")
            
            X = df_features[available_features].fillna(0)
            
            # Make prediction
            fraud_probability = self.model.predict_proba(X)[0, 1]
            is_fraud = fraud_probability > FRAUD_THRESHOLD
            
            # Determine risk level
            if fraud_probability >= HIGH_RISK_THRESHOLD:
                risk_level = "HIGH"
            elif fraud_probability >= FRAUD_THRESHOLD:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Generate explanation
            explanation = self._generate_explanation(transaction_data, df_features, fraud_probability)
            
            result = {
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_probability),
                'risk_level': risk_level,
                'threshold_used': FRAUD_THRESHOLD,
                'explanation': explanation,
                'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                'timestamp_processed': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'is_fraud': None,
                'fraud_probability': None,
                'risk_level': 'ERROR'
            }
    
    def predict_batch(self, transactions_df):
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions_df: DataFrame with transaction data
        
        Returns:
            DataFrame: Predictions for all transactions
        """
        if not self.is_ready:
            raise ValueError("Model not loaded. Call load_model first.")
        
        try:
            # Engineer features
            df_features = self.feature_engineer.transform(transactions_df.copy())
            
            # Get feature columns for model
            feature_columns = self.feature_engineer.get_feature_columns()
            available_features = [col for col in feature_columns if col in df_features.columns]
            
            X = df_features[available_features].fillna(0)
            
            # Make predictions
            fraud_probabilities = self.model.predict_proba(X)[:, 1]
            is_fraud = fraud_probabilities > FRAUD_THRESHOLD
            
            # Add predictions to dataframe
            results_df = transactions_df.copy()
            results_df['fraud_probability'] = fraud_probabilities
            results_df['is_fraud'] = is_fraud
            results_df['risk_level'] = pd.cut(
                fraud_probabilities,
                bins=[0, FRAUD_THRESHOLD, HIGH_RISK_THRESHOLD, 1.0],
                labels=['LOW', 'MEDIUM', 'HIGH'],
                include_lowest=True
            )
            
            return results_df
            
        except Exception as e:
            print(f"Error in batch prediction: {str(e)}")
            return None
    
    def _generate_explanation(self, transaction_data, features_df, fraud_prob):
        """Generate human-readable explanation for the prediction."""
        explanations = []
        
        amount = transaction_data.get('amount', 0)
        merchant = transaction_data.get('merchant_category', 'unknown')
        location_risk = transaction_data.get('location_risk_score', 0)
        
        # Amount-based explanations
        if amount > 1000:
            explanations.append(f"Large transaction amount (${amount:.2f})")
        elif amount < 5:
            explanations.append(f"Unusually small amount (${amount:.2f})")
        
        # Time-based explanations
        timestamp = transaction_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            hour = timestamp.hour
            if hour >= 23 or hour <= 6:
                explanations.append("Transaction during unusual hours (night)")
        
        # Merchant explanations
        risky_merchants = ['online_retail', 'electronics', 'jewelry']
        if merchant in risky_merchants:
            explanations.append(f"High-risk merchant category ({merchant})")
        
        # Location explanations
        if location_risk > 0.6:
            explanations.append("High-risk location")
        
        # Feature-based explanations (if available)
        if 'velocity_1h_scaled' in features_df.columns:
            velocity = features_df['velocity_1h_scaled'].iloc[0]
            if velocity > 2:
                explanations.append("High transaction frequency")
        
        if not explanations:
            if fraud_prob > 0.5:
                explanations.append("Multiple risk factors detected")
            else:
                explanations.append("Normal transaction pattern")
        
        return explanations
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if not self.is_ready:
            return {"status": "Model not loaded"}
        
        info = {
            "status": "Ready",
            "model_type": getattr(self.model, 'model_type', 'ensemble'),
            "fraud_threshold": FRAUD_THRESHOLD,
            "high_risk_threshold": HIGH_RISK_THRESHOLD
        }
        
        if hasattr(self.model, 'models'):
            # Ensemble model
            info["ensemble_models"] = [m.model_type for m in self.model.models]
        
        return info

class RealTimeMonitor:
    """Monitor for real-time fraud detection alerts."""
    
    def __init__(self, predictor):
        """Initialize monitor with a fraud predictor."""
        self.predictor = predictor
        self.alerts = []
        self.statistics = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'high_risk_transactions': 0,
            'avg_fraud_probability': 0.0
        }
    
    def process_transaction(self, transaction_data):
        """Process a transaction and generate alerts if needed."""
        result = self.predictor.predict_single_transaction(transaction_data)
        
        # Update statistics
        self.statistics['total_transactions'] += 1
        
        if result.get('is_fraud'):
            self.statistics['fraud_detected'] += 1
            
        if result.get('risk_level') == 'HIGH':
            self.statistics['high_risk_transactions'] += 1
        
        # Update average fraud probability
        if result.get('fraud_probability') is not None:
            total = self.statistics['total_transactions']
            current_avg = self.statistics['avg_fraud_probability']
            new_prob = result['fraud_probability']
            self.statistics['avg_fraud_probability'] = ((current_avg * (total - 1)) + new_prob) / total
        
        # Generate alert for high-risk transactions
        if result.get('risk_level') in ['HIGH', 'MEDIUM'] and result.get('is_fraud'):
            alert = {
                'timestamp': datetime.now().isoformat(),
                'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                'user_id': transaction_data.get('user_id', 'unknown'),
                'amount': transaction_data.get('amount', 0),
                'fraud_probability': result['fraud_probability'],
                'risk_level': result['risk_level'],
                'explanation': result['explanation']
            }
            self.alerts.append(alert)
        
        return result
    
    def get_alerts(self, limit=10):
        """Get recent fraud alerts."""
        return self.alerts[-limit:]
    
    def get_statistics(self):
        """Get monitoring statistics."""
        stats = self.statistics.copy()
        if stats['total_transactions'] > 0:
            stats['fraud_rate'] = stats['fraud_detected'] / stats['total_transactions']
            stats['high_risk_rate'] = stats['high_risk_transactions'] / stats['total_transactions']
        else:
            stats['fraud_rate'] = 0.0
            stats['high_risk_rate'] = 0.0
        
        return stats
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
        print("Alerts cleared.")
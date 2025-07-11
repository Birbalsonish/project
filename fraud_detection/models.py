"""
Machine learning models for credit card fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import *

class FraudDetectionModel:
    def __init__(self, model_type='random_forest', handle_imbalance=True):
        """
        Initialize fraud detection model.
        
        Args:
            model_type: Type of model ('random_forest', 'logistic', 'isolation_forest')
            handle_imbalance: Whether to handle class imbalance with class_weight
        """
        self.model_type = model_type
        self.handle_imbalance = handle_imbalance
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
        
        # Initialize the base model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model based on type."""
        if self.model_type == 'random_forest':
            base_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
            if self.handle_imbalance:
                base_model.set_params(class_weight='balanced')
        elif self.model_type == 'logistic':
            base_model = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced' if self.handle_imbalance else None
            )
        elif self.model_type == 'isolation_forest':
            base_model = IsolationForest(
                contamination=0.02,  # Estimated fraud rate
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = base_model
    
    def fit(self, X, y, feature_columns=None):
        """
        Train the fraud detection model.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_columns: List of feature column names
        """
        print(f"Training {self.model_type} model...")
        
        # Store feature columns
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Handle NaN values
        X = pd.DataFrame(X, columns=self.feature_columns).fillna(0)
        
        if self.model_type == 'isolation_forest':
            # Isolation Forest is unsupervised
            self.model.fit(X)
        else:
            # Supervised learning
            self.model.fit(X, y)
        
        self.is_fitted = True
        print(f"Model training completed. Features: {len(self.feature_columns)}")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        # Handle NaN values
        X = pd.DataFrame(X, columns=self.feature_columns).fillna(0)
        
        if self.model_type == 'isolation_forest':
            # Isolation Forest returns -1 for outliers, 1 for inliers
            predictions = self.model.predict(X)
            return (predictions == -1).astype(int)  # Convert to 0/1
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        if self.model_type == 'isolation_forest':
            # Isolation Forest doesn't have predict_proba
            # Use decision_function as a proxy
            scores = self.model.decision_function(X)
            # Convert to probabilities (higher negative score = more anomalous)
            proba = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            return np.column_stack([1 - proba, proba])
        else:
            # Handle NaN values
            X = pd.DataFrame(X, columns=self.feature_columns).fillna(0)
            return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance.")
        
        if self.model_type == 'isolation_forest':
            return None  # Isolation Forest doesn't have feature importance
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'handle_imbalance': self.handle_imbalance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            model_type=model_data['model_type'],
            handle_imbalance=model_data['handle_imbalance']
        )
        
        # Load model components
        instance.model = model_data['model']
        instance.feature_columns = model_data['feature_columns']
        instance.is_fitted = True
        
        return instance

class EnsembleModel:
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models=None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of (model_type, weight) tuples
        """
        if models is None:
            models = [
                ('random_forest', 0.5),
                ('logistic', 0.5)
            ]
        
        self.models = []
        self.weights = []
        
        for model_type, weight in models:
            model = FraudDetectionModel(model_type=model_type, handle_imbalance=True)
            self.models.append(model)
            self.weights.append(weight)
        
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()  # Normalize weights
        self.is_fitted = False
    
    def fit(self, X, y, feature_columns=None):
        """Train all models in the ensemble."""
        print("Training ensemble models...")
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.model_type}")
            model.fit(X, y, feature_columns)
        
        self.is_fitted = True
        print("Ensemble training completed.")
        
        return self
    
    def predict_proba(self, X):
        """Get ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions.")
        
        predictions = []
        for model in self.models:
            proba = model.predict_proba(X)
            predictions.append(proba)
        
        # Weighted average of predictions
        ensemble_proba = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_proba
    
    def predict(self, X, threshold=FRAUD_THRESHOLD):
        """Make ensemble predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)
    
    def save_ensemble(self, filepath):
        """Save the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving.")
        
        ensemble_data = {
            'models': self.models,
            'weights': self.weights
        }
        
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath):
        """Load an ensemble model."""
        ensemble_data = joblib.load(filepath)
        
        instance = cls()
        instance.models = ensemble_data['models']
        instance.weights = ensemble_data['weights']
        instance.is_fitted = True
        
        return instance

def evaluate_model(model, X_test, y_test, threshold=FRAUD_THRESHOLD):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
    
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model performance...")
    
    # Get predictions and probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'roc_auc': roc_auc,
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'accuracy': class_report['accuracy'],
        'confusion_matrix': cm,
        'classification_report': class_report,
        'precision_curve': precision,
        'recall_curve': recall,
        'pr_thresholds': thresholds
    }
    
    print(f"Model Performance:")
    print(f"- ROC AUC: {roc_auc:.4f}")
    print(f"- Precision: {results['precision']:.4f}")
    print(f"- Recall: {results['recall']:.4f}")
    print(f"- F1 Score: {results['f1_score']:.4f}")
    print(f"- Accuracy: {results['accuracy']:.4f}")
    
    return results
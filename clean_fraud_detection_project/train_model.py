#!/usr/bin/env python3
"""
Main training script for credit card fraud detection system.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import argparse
from datetime import datetime

# Import fraud detection modules
from fraud_detection.data_generator import CreditCardDataGenerator
from fraud_detection.feature_engineering import FeatureEngineer
from fraud_detection.models import FraudDetectionModel, EnsembleModel, evaluate_model
from fraud_detection.config import *

def create_directories():
    """Create necessary directories."""
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    print("Directories created successfully.")

def generate_training_data(n_transactions=50000, fraud_rate=0.02):
    """Generate synthetic training data."""
    print("=" * 60)
    print("STEP 1: GENERATING TRAINING DATA")
    print("=" * 60)
    
    data_file = os.path.join(DATA_PATH, "credit_card_transactions.csv")
    
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} transactions")
    else:
        print("Generating new synthetic data...")
        generator = CreditCardDataGenerator()
        df = generator.generate_dataset(
            n_transactions=n_transactions,
            fraud_rate=fraud_rate,
            n_users=min(2000, n_transactions // 10)
        )
        
        # Save generated data
        df.to_csv(data_file, index=False)
        print(f"Data saved to {data_file}")
    
    return df

def engineer_features(df):
    """Engineer features from raw transaction data."""
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.fit_transform(df)
    
    # Save feature engineer
    feature_engineer_path = os.path.join(MODELS_PATH, "feature_engineer.pkl")
    joblib.dump(feature_engineer, feature_engineer_path)
    print(f"Feature engineer saved to {feature_engineer_path}")
    
    # Get feature columns for modeling
    feature_columns = feature_engineer.get_feature_columns()
    available_features = [col for col in feature_columns if col in df_features.columns]
    
    print(f"Available features for modeling: {len(available_features)}")
    print("Feature columns:", available_features[:10], "..." if len(available_features) > 10 else "")
    
    return df_features, available_features, feature_engineer

def prepare_training_data(df_features, feature_columns):
    """Prepare data for training."""
    print("\n" + "=" * 60)
    print("STEP 3: PREPARING TRAINING DATA")
    print("=" * 60)
    
    # Select features and target
    X = df_features[feature_columns].fillna(0)
    y = df_features['is_fraud']
    
    print(f"Training data shape: {X.shape}")
    print(f"Class distribution:")
    print(f"- Normal transactions: {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"- Fraudulent transactions: {(y == 1).sum()} ({(y == 1).mean():.1%})")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VALIDATION_SIZE, 
        random_state=RANDOM_STATE, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VALIDATION_SIZE),
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"- Training: {X_train.shape[0]} samples")
    print(f"- Validation: {X_val.shape[0]} samples")
    print(f"- Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, y_train, X_val, y_val, feature_columns):
    """Train different models and select the best one."""
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING MODELS")
    print("=" * 60)
    
    # Models to train (only available models)
    model_configs = [
        ('random_forest', True),
        ('logistic', True)
    ]
    
    models = {}
    validation_scores = {}
    
    # Train individual models
    for model_type, handle_imbalance in model_configs:
        print(f"\nTraining {model_type} model...")
        
        model = FraudDetectionModel(
            model_type=model_type,
            handle_imbalance=handle_imbalance
        )
        
        model.fit(X_train, y_train, feature_columns)
        
        # Evaluate on validation set
        val_results = evaluate_model(model, X_val, y_val)
        models[model_type] = model
        validation_scores[model_type] = val_results
        
        # Save individual model
        model_path = os.path.join(MODELS_PATH, f"{model_type}_model.pkl")
        model.save_model(model_path)
    
    # Train ensemble model
    print(f"\nTraining ensemble model...")
    ensemble_model = EnsembleModel()
    ensemble_model.fit(X_train, y_train, feature_columns)
    
    # Evaluate ensemble
    ensemble_results = evaluate_model(ensemble_model, X_val, y_val)
    models['ensemble'] = ensemble_model
    validation_scores['ensemble'] = ensemble_results
    
    # Save ensemble model
    ensemble_path = os.path.join(MODELS_PATH, "ensemble_model.pkl")
    ensemble_model.save_ensemble(ensemble_path)
    
    return models, validation_scores

def select_best_model(models, validation_scores):
    """Select the best model based on validation performance."""
    print("\n" + "=" * 60)
    print("STEP 5: MODEL SELECTION")
    print("=" * 60)
    
    print("Validation Performance Summary:")
    print("-" * 80)
    print(f"{'Model':<15} {'ROC-AUC':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
    print("-" * 80)
    
    best_model_name = None
    best_score = 0
    
    for model_name, scores in validation_scores.items():
        roc_auc = scores['roc_auc']
        precision = scores['precision']
        recall = scores['recall']
        f1 = scores['f1_score']
        
        print(f"{model_name:<15} {roc_auc:<8.4f} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f}")
        
        # Use F1 score as primary metric for selection
        if f1 > best_score:
            best_score = f1
            best_model_name = model_name
    
    print("-" * 80)
    print(f"Best model: {best_model_name} (F1-Score: {best_score:.4f})")
    
    return models[best_model_name], best_model_name

def final_evaluation(best_model, X_test, y_test, model_name):
    """Perform final evaluation on test set."""
    print("\n" + "=" * 60)
    print("STEP 6: FINAL EVALUATION")
    print("=" * 60)
    
    print(f"Evaluating {model_name} on test set...")
    test_results = evaluate_model(best_model, X_test, y_test)
    
    # Feature importance analysis
    if hasattr(best_model, 'get_feature_importance'):
        importance_df = best_model.get_feature_importance()
        if importance_df is not None:
            print(f"\nTop 10 Most Important Features:")
            print("-" * 40)
            for idx, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    return test_results

def save_training_report(validation_scores, test_results, model_name):
    """Save training report."""
    print("\n" + "=" * 60)
    print("STEP 7: SAVING TRAINING REPORT")
    print("=" * 60)
    
    report = {
        'training_timestamp': datetime.now().isoformat(),
        'best_model': model_name,
        'validation_scores': validation_scores,
        'test_results': test_results,
        'model_config': {
            'fraud_threshold': FRAUD_THRESHOLD,
            'high_risk_threshold': HIGH_RISK_THRESHOLD,
            'random_state': RANDOM_STATE
        }
    }
    
    report_path = os.path.join(LOGS_PATH, "training_report.json")
    
    import json
    with open(report_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Clean the report for JSON serialization
        clean_report = {}
        for key, value in report.items():
            if key == 'test_results':
                clean_value = {}
                for k, v in value.items():
                    if k not in ['confusion_matrix', 'precision_curve', 'recall_curve', 'pr_thresholds']:
                        clean_value[k] = convert_numpy(v)
                clean_report[key] = clean_value
            else:
                clean_report[key] = value
        
        json.dump(clean_report, f, indent=2, default=convert_numpy)
    
    print(f"Training report saved to {report_path}")

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Credit Card Fraud Detection Model')
    parser.add_argument('--transactions', type=int, default=50000,
                       help='Number of transactions to generate (default: 50000)')
    parser.add_argument('--fraud-rate', type=float, default=0.02,
                       help='Fraud rate in generated data (default: 0.02)')
    parser.add_argument('--skip-data-generation', action='store_true',
                       help='Skip data generation and use existing data')
    
    args = parser.parse_args()
    
    print("🔍 CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Transactions: {args.transactions:,}")
    print(f"Fraud Rate: {args.fraud_rate:.1%}")
    print(f"Random State: {RANDOM_STATE}")
    print("=" * 60)
    
    try:
        # Create directories
        create_directories()
        
        # Generate or load data
        if not args.skip_data_generation:
            df = generate_training_data(args.transactions, args.fraud_rate)
        else:
            data_file = os.path.join(DATA_PATH, "credit_card_transactions.csv")
            df = pd.read_csv(data_file)
            print(f"Loaded existing data: {len(df)} transactions")
        
        # Feature engineering
        df_features, feature_columns, feature_engineer = engineer_features(df)
        
        # Prepare training data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(
            df_features, feature_columns
        )
        
        # Train models
        models, validation_scores = train_models(
            X_train, y_train, X_val, y_val, feature_columns
        )
        
        # Select best model
        best_model, model_name = select_best_model(models, validation_scores)
        
        # Final evaluation
        test_results = final_evaluation(best_model, X_test, y_test, model_name)
        
        # Save training report
        save_training_report(validation_scores, test_results, model_name)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Best model: {model_name}")
        print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
        print(f"Test F1-Score: {test_results['f1_score']:.4f}")
        print("\nModel artifacts saved in:", MODELS_PATH)
        print("Training logs saved in:", LOGS_PATH)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
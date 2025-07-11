#!/usr/bin/env python3
"""
Flask API server for real-time credit card fraud detection.
"""

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd

from fraud_detection.predictor import FraudPredictor, RealTimeMonitor
from fraud_detection.config import API_HOST, API_PORT, DEBUG, MODELS_PATH

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global predictor and monitor
predictor = None
monitor = None

def initialize_predictor():
    """Initialize the fraud predictor with trained models."""
    global predictor, monitor
    
    try:
        # Look for available models
        model_files = {
            'ensemble': os.path.join(MODELS_PATH, 'ensemble_model.pkl'),
            'xgboost': os.path.join(MODELS_PATH, 'xgboost_model.pkl'),
            'random_forest': os.path.join(MODELS_PATH, 'random_forest_model.pkl'),
            'lightgbm': os.path.join(MODELS_PATH, 'lightgbm_model.pkl')
        }
        
        feature_engineer_path = os.path.join(MODELS_PATH, 'feature_engineer.pkl')
        
        # Find the first available model
        model_path = None
        model_type = None
        for mtype, mpath in model_files.items():
            if os.path.exists(mpath):
                model_path = mpath
                model_type = mtype
                break
        
        if model_path and os.path.exists(feature_engineer_path):
            predictor = FraudPredictor(model_path, feature_engineer_path)
            monitor = RealTimeMonitor(predictor)
            print(f"✅ Fraud detection system initialized with {model_type} model")
            return True
        else:
            print("❌ No trained models found. Please run train_model.py first.")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing predictor: {str(e)}")
        return False

@app.route('/')
def home():
    """Home page with API documentation."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Card Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .method { background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .status { padding: 10px; border-radius: 5px; margin: 20px 0; }
            .ready { background: #d5f4e6; border: 1px solid #27ae60; color: #27ae60; }
            .error { background: #fadbd8; border: 1px solid #e74c3c; color: #e74c3c; }
            code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .example { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Credit Card Fraud Detection API</h1>
            
            <div class="status {{ status_class }}">
                <strong>System Status:</strong> {{ status_message }}
            </div>
            
            <h2>📋 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code>
                <p>Check system health and model status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict</code>
                <p>Predict fraud for a single transaction</p>
                <div class="example">
                    <strong>Example Request:</strong>
                    <pre>{
  "user_id": "user_123",
  "amount": 150.75,
  "merchant_category": "grocery",
  "location_risk_score": 0.2,
  "timestamp": "2024-01-15T14:30:00"
}</pre>
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/predict_batch</code>
                <p>Predict fraud for multiple transactions</p>
                <div class="example">
                    <strong>Example Request:</strong>
                    <pre>{
  "transactions": [
    {
      "user_id": "user_123",
      "amount": 150.75,
      "merchant_category": "grocery",
      "location_risk_score": 0.2
    },
    {
      "user_id": "user_456",
      "amount": 2500.00,
      "merchant_category": "electronics",
      "location_risk_score": 0.8
    }
  ]
}</pre>
                </div>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/alerts</code>
                <p>Get recent fraud alerts</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/statistics</code>
                <p>Get monitoring statistics</p>
            </div>
            
            <h2>📊 Usage Example</h2>
            <p>To test the API with curl:</p>
            <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "user_123",
    "amount": 150.75,
    "merchant_category": "grocery",
    "location_risk_score": 0.2
  }'</pre>
            
            <h2>📈 Response Format</h2>
            <pre>{
  "is_fraud": false,
  "fraud_probability": 0.15,
  "risk_level": "LOW",
  "explanation": ["Normal transaction pattern"],
  "transaction_id": "unknown",
  "timestamp_processed": "2024-01-15T14:30:00.123456"
}</pre>
        </div>
    </body>
    </html>
    """
    
    if predictor and predictor.is_ready:
        status_class = "ready"
        status_message = "System Ready - Models Loaded Successfully"
    else:
        status_class = "error"
        status_message = "System Not Ready - Please train models first (run train_model.py)"
    
    return render_template_string(html_template, 
                                status_class=status_class, 
                                status_message=status_message)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if predictor and predictor.is_ready:
        model_info = predictor.get_model_info()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': 'Model not loaded'
        }), 503

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict fraud for a single transaction."""
    if not predictor or not predictor.is_ready:
        return jsonify({
            'error': 'Model not loaded. Please check system health.'
        }), 503
    
    try:
        transaction_data = request.json
        
        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in transaction_data:
            transaction_data['timestamp'] = datetime.now().isoformat()
        
        # Make prediction
        result = monitor.process_transaction(transaction_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict fraud for multiple transactions."""
    if not predictor or not predictor.is_ready:
        return jsonify({
            'error': 'Model not loaded. Please check system health.'
        }), 503
    
    try:
        data = request.json
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions data provided'}), 400
        
        transactions = data['transactions']
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Add timestamps if not provided
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        # Make predictions
        results_df = predictor.predict_batch(df)
        
        if results_df is not None:
            # Convert to list of dictionaries
            results = results_df.to_dict('records')
            
            # Process each transaction through monitor for alerts
            for i, transaction in enumerate(transactions):
                if i < len(results):
                    transaction_with_result = transaction.copy()
                    transaction_with_result.update(results[i])
                    monitor.process_transaction(transaction_with_result)
            
            return jsonify({
                'predictions': results,
                'total_transactions': len(results),
                'fraud_detected': sum(1 for r in results if r.get('is_fraud', False))
            })
        else:
            return jsonify({'error': 'Batch prediction failed'}), 500
            
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get recent fraud alerts."""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 503
    
    try:
        limit = request.args.get('limit', 10, type=int)
        alerts = monitor.get_alerts(limit)
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get alerts: {str(e)}'
        }), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get monitoring statistics."""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 503
    
    try:
        stats = monitor.get_statistics()
        
        return jsonify({
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get statistics: {str(e)}'
        }), 500

@app.route('/clear_alerts', methods=['POST'])
def clear_alerts():
    """Clear all alerts."""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'}), 503
    
    try:
        monitor.clear_alerts()
        return jsonify({
            'message': 'Alerts cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to clear alerts: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Credit Card Fraud Detection API Server...")
    print("=" * 60)
    
    # Initialize predictor
    if initialize_predictor():
        print(f"Server starting at http://{API_HOST}:{API_PORT}")
        print("API Documentation: http://localhost:5000")
        print("Health Check: http://localhost:5000/health")
        print("=" * 60)
        
        app.run(
            host=API_HOST,
            port=API_PORT,
            debug=DEBUG
        )
    else:
        print("❌ Failed to initialize fraud detection system.")
        print("Please run 'python train_model.py' first to train the models.")
        exit(1)
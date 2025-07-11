#!/usr/bin/env python3
"""
Demo script for credit card fraud detection system.
"""

import os
import requests
import json
import time
from datetime import datetime, timedelta
import random

def test_api_connection():
    """Test if the API server is running."""
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print("❌ API server is not healthy")
            return False
    except requests.exceptions.RequestException:
        print("❌ API server is not running")
        print("Please run: python api_server.py")
        return False

def generate_demo_transactions():
    """Generate sample transactions for demonstration."""
    transactions = [
        # Normal transactions
        {
            "transaction_id": "txn_001",
            "user_id": "user_123",
            "amount": 45.67,
            "merchant_category": "grocery",
            "location_risk_score": 0.1,
            "timestamp": datetime.now().isoformat()
        },
        {
            "transaction_id": "txn_002", 
            "user_id": "user_456",
            "amount": 125.50,
            "merchant_category": "restaurant",
            "location_risk_score": 0.2,
            "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
        },
        {
            "transaction_id": "txn_003",
            "user_id": "user_789",
            "amount": 75.20,
            "merchant_category": "gas_station",
            "location_risk_score": 0.15,
            "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
        },
        
        # Potentially fraudulent transactions
        {
            "transaction_id": "txn_004",
            "user_id": "user_999",
            "amount": 2500.00,
            "merchant_category": "electronics",
            "location_risk_score": 0.8,
            "timestamp": (datetime.now() - timedelta(minutes=2)).isoformat()
        },
        {
            "transaction_id": "txn_005",
            "user_id": "user_888",
            "amount": 1899.99,
            "merchant_category": "jewelry",
            "location_risk_score": 0.9,
            "timestamp": (datetime.now() - timedelta(minutes=1)).isoformat()
        },
        {
            "transaction_id": "txn_006",
            "user_id": "user_777",
            "amount": 5.99,
            "merchant_category": "online_retail",
            "location_risk_score": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return transactions

def test_single_prediction(transaction):
    """Test single transaction prediction."""
    try:
        response = requests.post(
            'http://localhost:5000/predict',
            json=transaction,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ Error predicting transaction: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return None

def test_batch_prediction(transactions):
    """Test batch transaction prediction."""
    try:
        response = requests.post(
            'http://localhost:5000/predict_batch',
            json={'transactions': transactions},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ Error in batch prediction: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch request failed: {str(e)}")
        return None

def get_system_statistics():
    """Get system monitoring statistics."""
    try:
        response = requests.get('http://localhost:5000/statistics', timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error getting statistics: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Statistics request failed: {str(e)}")
        return None

def get_fraud_alerts():
    """Get recent fraud alerts."""
    try:
        response = requests.get('http://localhost:5000/alerts', timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error getting alerts: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Alerts request failed: {str(e)}")
        return None

def format_prediction_result(transaction, result):
    """Format prediction result for display."""
    if not result:
        return f"❌ Failed to process transaction {transaction.get('transaction_id', 'unknown')}"
    
    fraud_status = "🚨 FRAUD DETECTED" if result.get('is_fraud') else "✅ Normal"
    risk_level = result.get('risk_level', 'Unknown')
    probability = result.get('fraud_probability', 0) * 100
    
    output = f"""
Transaction ID: {transaction.get('transaction_id', 'unknown')}
User: {transaction.get('user_id', 'unknown')}
Amount: ${transaction.get('amount', 0):,.2f}
Merchant: {transaction.get('merchant_category', 'unknown')}
Status: {fraud_status}
Risk Level: {risk_level}
Fraud Probability: {probability:.1f}%
Explanation: {', '.join(result.get('explanation', ['No explanation']))}
"""
    return output

def main():
    """Run the fraud detection demo."""
    print("🔍 CREDIT CARD FRAUD DETECTION - DEMO")
    print("=" * 60)
    
    # Test API connection
    if not test_api_connection():
        return 1
    
    # Generate demo transactions
    transactions = generate_demo_transactions()
    
    print(f"\n📊 Generated {len(transactions)} demo transactions")
    print("=" * 60)
    
    # Test single predictions
    print("\n🎯 TESTING SINGLE PREDICTIONS")
    print("-" * 40)
    
    for i, transaction in enumerate(transactions[:3]):
        print(f"\nTesting transaction {i+1}/3...")
        result = test_single_prediction(transaction)
        
        if result:
            print(format_prediction_result(transaction, result))
        
        time.sleep(1)  # Brief pause between requests
    
    # Test batch prediction
    print("\n📦 TESTING BATCH PREDICTION")
    print("-" * 40)
    
    batch_result = test_batch_prediction(transactions)
    
    if batch_result:
        predictions = batch_result.get('predictions', [])
        total_transactions = batch_result.get('total_transactions', 0)
        fraud_detected = batch_result.get('fraud_detected', 0)
        
        print(f"\nBatch Results:")
        print(f"Total Transactions: {total_transactions}")
        print(f"Fraud Detected: {fraud_detected}")
        print(f"Fraud Rate: {fraud_detected/total_transactions*100:.1f}%")
        
        print(f"\nDetailed Results:")
        for i, (transaction, prediction) in enumerate(zip(transactions, predictions)):
            fraud_status = "🚨 FRAUD" if prediction.get('is_fraud') else "✅ Normal"
            probability = prediction.get('fraud_probability', 0) * 100
            risk_level = prediction.get('risk_level', 'Unknown')
            
            print(f"{i+1}. {transaction['transaction_id']}: {fraud_status} "
                  f"({probability:.1f}%, {risk_level})")
    
    # Get system statistics
    print("\n📈 SYSTEM STATISTICS")
    print("-" * 40)
    
    stats = get_system_statistics()
    if stats:
        statistics = stats.get('statistics', {})
        print(f"Total Transactions Processed: {statistics.get('total_transactions', 0)}")
        print(f"Fraud Detected: {statistics.get('fraud_detected', 0)}")
        print(f"High Risk Transactions: {statistics.get('high_risk_transactions', 0)}")
        print(f"Average Fraud Probability: {statistics.get('avg_fraud_probability', 0)*100:.1f}%")
        print(f"Overall Fraud Rate: {statistics.get('fraud_rate', 0)*100:.1f}%")
    
    # Get fraud alerts
    print("\n🚨 RECENT FRAUD ALERTS")
    print("-" * 40)
    
    alerts = get_fraud_alerts()
    if alerts and alerts.get('alerts'):
        for i, alert in enumerate(alerts['alerts'][:5]):  # Show last 5 alerts
            print(f"{i+1}. Transaction {alert.get('transaction_id', 'unknown')} "
                  f"- User {alert.get('user_id', 'unknown')} "
                  f"- ${alert.get('amount', 0):,.2f} "
                  f"- {alert.get('risk_level', 'Unknown')} Risk "
                  f"({alert.get('fraud_probability', 0)*100:.1f}%)")
    else:
        print("No recent fraud alerts")
    
    # Performance testing
    print("\n⚡ PERFORMANCE TEST")
    print("-" * 40)
    
    start_time = time.time()
    test_transactions = []
    
    # Generate more test transactions
    for i in range(20):
        test_transaction = {
            "transaction_id": f"perf_test_{i+1}",
            "user_id": f"user_{random.randint(1, 100)}",
            "amount": round(random.uniform(10, 1000), 2),
            "merchant_category": random.choice(['grocery', 'restaurant', 'electronics', 'gas_station']),
            "location_risk_score": round(random.uniform(0.1, 0.8), 2),
            "timestamp": datetime.now().isoformat()
        }
        test_transactions.append(test_transaction)
    
    # Test batch performance
    batch_result = test_batch_prediction(test_transactions)
    end_time = time.time()
    
    if batch_result:
        processing_time = end_time - start_time
        transactions_per_second = len(test_transactions) / processing_time
        
        print(f"Processed {len(test_transactions)} transactions in {processing_time:.2f} seconds")
        print(f"Performance: {transactions_per_second:.1f} transactions/second")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("You can now:")
    print("- Visit http://localhost:5000 for API documentation")
    print("- Use the API endpoints for real-time fraud detection")
    print("- Monitor alerts and statistics through the API")
    
    return 0

if __name__ == "__main__":
    exit(main())
"""
Synthetic credit card transaction data generator for fraud detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class CreditCardDataGenerator:
    def __init__(self, random_state=42):
        """Initialize the data generator."""
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Merchant categories
        self.merchant_categories = [
            'grocery', 'gas_station', 'restaurant', 'online_retail', 
            'department_store', 'pharmacy', 'entertainment', 'hotel',
            'airline', 'car_rental', 'electronics', 'jewelry'
        ]
        
        # Location risk scores (higher = more risky)
        self.location_risk_map = {
            'domestic_low': 0.1,
            'domestic_medium': 0.3,
            'domestic_high': 0.6,
            'international_low': 0.4,
            'international_medium': 0.7,
            'international_high': 0.9
        }
    
    def generate_normal_transaction(self, user_id, timestamp):
        """Generate a normal (non-fraudulent) transaction."""
        # Normal transaction patterns
        amount = np.random.lognormal(mean=3, sigma=1.5)  # Most transactions are small
        amount = max(5, min(amount, 2000))  # Between $5 and $2000
        
        # Merchant category (some categories more common)
        merchant_weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02]
        merchant_category = np.random.choice(self.merchant_categories, p=merchant_weights)
        
        # Location (mostly domestic, low risk)
        location_weights = [0.6, 0.25, 0.1, 0.03, 0.015, 0.005]
        location = np.random.choice(list(self.location_risk_map.keys()), p=location_weights)
        location_risk_score = self.location_risk_map[location]
        
        return {
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'location_risk_score': location_risk_score,
            'is_fraud': 0
        }
    
    def generate_fraudulent_transaction(self, user_id, timestamp):
        """Generate a fraudulent transaction."""
        # Fraudulent transaction patterns
        if random.random() < 0.3:
            # Large amount fraud
            amount = np.random.uniform(500, 5000)
        else:
            # Small amount fraud (to avoid detection)
            amount = np.random.uniform(1, 100)
        
        # Fraudulent transactions often in risky categories
        risky_categories = ['online_retail', 'electronics', 'jewelry', 'entertainment']
        merchant_category = random.choice(risky_categories)
        
        # Higher location risk
        risky_locations = ['international_medium', 'international_high', 'domestic_high']
        location_weights = [0.4, 0.4, 0.2]
        location = np.random.choice(risky_locations, p=location_weights)
        location_risk_score = self.location_risk_map[location]
        
        return {
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'location_risk_score': location_risk_score,
            'is_fraud': 1
        }
    
    def generate_dataset(self, n_transactions=10000, fraud_rate=0.02, n_users=1000):
        """Generate a complete dataset with normal and fraudulent transactions."""
        print(f"Generating {n_transactions} transactions for {n_users} users...")
        
        transactions = []
        n_fraud = int(n_transactions * fraud_rate)
        n_normal = n_transactions - n_fraud
        
        # Generate timestamps over the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Generate normal transactions
        for i in range(n_normal):
            user_id = f"user_{random.randint(1, n_users)}"
            timestamp = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            transaction = self.generate_normal_transaction(user_id, timestamp)
            transactions.append(transaction)
        
        # Generate fraudulent transactions
        for i in range(n_fraud):
            user_id = f"user_{random.randint(1, n_users)}"
            timestamp = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            transaction = self.generate_fraudulent_transaction(user_id, timestamp)
            transactions.append(transaction)
        
        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(transactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Generated dataset:")
        print(f"- Total transactions: {len(df)}")
        print(f"- Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.1%})")
        print(f"- Normal transactions: {len(df) - df['is_fraud'].sum()}")
        
        return df

def generate_sample_data():
    """Generate sample data for testing."""
    generator = CreditCardDataGenerator()
    return generator.generate_dataset(n_transactions=10000, fraud_rate=0.02)

if __name__ == "__main__":
    # Generate and save sample data
    df = generate_sample_data()
    df.to_csv("data/credit_card_transactions.csv", index=False)
    print("Sample data saved to data/credit_card_transactions.csv")
"""
Visualization tools for fraud detection analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class FraudAnalysisVisualizer:
    """Visualization tools for fraud detection analysis."""
    
    def __init__(self, figsize=(12, 8)):
        """Initialize the visualizer."""
        self.figsize = figsize
        self.color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.color_palette)
    
    def plot_class_distribution(self, y, title="Class Distribution"):
        """Plot the distribution of fraud vs normal transactions."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        counts = pd.Series(y).value_counts()
        labels = ['Normal', 'Fraud']
        colors = ['#2ecc71', '#e74c3c']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts.values, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0, 0.1)
        )
        
        # Add count information
        for i, (label, count) in enumerate(zip(labels, counts.values)):
            autotexts[i].set_text(f'{label}\n{count:,}\n({count/len(y)*100:.1f}%)')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot feature importance."""
        if importance_df is None:
            print("No feature importance data available")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Take top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(
            range(len(top_features)),
            top_features['importance'],
            color=plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true, y_scores, title="ROC Curve"):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = np.trapz(tpr, fpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='#e74c3c', linewidth=2, 
               label=f'ROC Curve (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='#95a5a6', linestyle='--', linewidth=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_scores, title="Precision-Recall Curve"):
        """Plot precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auc_score = np.trapz(precision, recall)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot PR curve
        ax.plot(recall, precision, color='#3498db', linewidth=2,
               label=f'PR Curve (AUC = {auc_score:.3f})')
        
        # Plot baseline
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='#95a5a6', linestyle='--', linewidth=1,
                  label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'],
                   ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add accuracy information
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
               transform=ax.transAxes, ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_transaction_patterns(self, df):
        """Plot transaction patterns and fraud distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Amount distribution by fraud status
        fraud_amounts = df[df['is_fraud'] == 1]['amount']
        normal_amounts = df[df['is_fraud'] == 0]['amount']
        
        axes[0,0].hist(normal_amounts, bins=50, alpha=0.7, label='Normal', 
                      color='#2ecc71', density=True)
        axes[0,0].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', 
                      color='#e74c3c', density=True)
        axes[0,0].set_xlabel('Transaction Amount')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Transaction Amount Distribution')
        axes[0,0].legend()
        axes[0,0].set_xlim(0, df['amount'].quantile(0.95))
        
        # 2. Fraud rate by merchant category
        fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].agg(['mean', 'count']).reset_index()
        fraud_by_merchant = fraud_by_merchant[fraud_by_merchant['count'] >= 10]  # Filter small categories
        
        bars = axes[0,1].bar(fraud_by_merchant['merchant_category'], 
                           fraud_by_merchant['mean'] * 100)
        axes[0,1].set_xlabel('Merchant Category')
        axes[0,1].set_ylabel('Fraud Rate (%)')
        axes[0,1].set_title('Fraud Rate by Merchant Category')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Color bars based on fraud rate
        for bar, rate in zip(bars, fraud_by_merchant['mean']):
            if rate > 0.05:  # High fraud rate
                bar.set_color('#e74c3c')
            elif rate > 0.02:  # Medium fraud rate
                bar.set_color('#f39c12')
            else:  # Low fraud rate
                bar.set_color('#2ecc71')
        
        # 3. Fraud by hour of day
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            fraud_by_hour = df.groupby('hour')['is_fraud'].agg(['mean', 'count']).reset_index()
            
            axes[1,0].plot(fraud_by_hour['hour'], fraud_by_hour['mean'] * 100, 
                          marker='o', linewidth=2, markersize=6)
            axes[1,0].set_xlabel('Hour of Day')
            axes[1,0].set_ylabel('Fraud Rate (%)')
            axes[1,0].set_title('Fraud Rate by Hour of Day')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xlim(0, 23)
        
        # 4. Location risk vs fraud rate
        if 'location_risk_score' in df.columns:
            df['risk_bin'] = pd.cut(df['location_risk_score'], bins=10)
            fraud_by_risk = df.groupby('risk_bin')['is_fraud'].agg(['mean', 'count']).reset_index()
            
            # Get bin centers for x-axis
            bin_centers = [interval.mid for interval in fraud_by_risk['risk_bin']]
            
            axes[1,1].scatter(bin_centers, fraud_by_risk['mean'] * 100, 
                            s=fraud_by_risk['count']*2, alpha=0.7)
            axes[1,1].set_xlabel('Location Risk Score')
            axes[1,1].set_ylabel('Fraud Rate (%)')
            axes[1,1].set_title('Fraud Rate vs Location Risk Score')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, validation_scores):
        """Plot comparison of different models."""
        if not validation_scores:
            print("No validation scores provided")
            return None
        
        models = list(validation_scores.keys())
        metrics = ['roc_auc', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [validation_scores[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color=self.color_palette[:len(models)])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Set y-axis limits for better visualization
            axes[i].set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, df, predictions=None):
        """Create an interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Distribution', 'Amount vs Risk Score', 
                          'Merchant Category Analysis', 'Time Series'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Fraud distribution pie chart
        fraud_counts = df['is_fraud'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Normal', 'Fraud'], 
                  values=[fraud_counts[0], fraud_counts[1]],
                  hole=0.3),
            row=1, col=1
        )
        
        # 2. Amount vs Risk Score scatter
        colors = ['red' if fraud else 'blue' for fraud in df['is_fraud']]
        fig.add_trace(
            go.Scatter(x=df['location_risk_score'], 
                      y=df['amount'],
                      mode='markers',
                      marker=dict(color=colors, size=8, opacity=0.6),
                      text=df['user_id'],
                      name='Transactions'),
            row=1, col=2
        )
        
        # 3. Merchant category analysis
        if 'merchant_category' in df.columns:
            fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=fraud_by_merchant['merchant_category'],
                      y=fraud_by_merchant['is_fraud'] * 100,
                      name='Fraud Rate %'),
                row=2, col=1
            )
        
        # 4. Time series (if timestamp available)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_fraud = df.groupby(df['timestamp'].dt.date)['is_fraud'].agg(['sum', 'count']).reset_index()
            daily_fraud['fraud_rate'] = daily_fraud['sum'] / daily_fraud['count'] * 100
            
            fig.add_trace(
                go.Scatter(x=daily_fraud['timestamp'],
                          y=daily_fraud['fraud_rate'],
                          mode='lines+markers',
                          name='Daily Fraud Rate %'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Credit Card Fraud Detection Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_all_plots(self, output_dir, **kwargs):
        """Save all generated plots to a directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plots_saved = []
        
        # Save each plot if data is provided
        if 'y_true' in kwargs and 'y_scores' in kwargs:
            # ROC Curve
            roc_fig = self.plot_roc_curve(kwargs['y_true'], kwargs['y_scores'])
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            roc_fig.savefig(roc_path, dpi=300, bbox_inches='tight')
            plots_saved.append(roc_path)
            plt.close(roc_fig)
            
            # Precision-Recall Curve
            pr_fig = self.plot_precision_recall_curve(kwargs['y_true'], kwargs['y_scores'])
            pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
            pr_fig.savefig(pr_path, dpi=300, bbox_inches='tight')
            plots_saved.append(pr_path)
            plt.close(pr_fig)
        
        if 'y_true' in kwargs and 'y_pred' in kwargs:
            # Confusion Matrix
            cm_fig = self.plot_confusion_matrix(kwargs['y_true'], kwargs['y_pred'])
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
            plots_saved.append(cm_path)
            plt.close(cm_fig)
        
        if 'importance_df' in kwargs:
            # Feature Importance
            fi_fig = self.plot_feature_importance(kwargs['importance_df'])
            if fi_fig:
                fi_path = os.path.join(output_dir, 'feature_importance.png')
                fi_fig.savefig(fi_path, dpi=300, bbox_inches='tight')
                plots_saved.append(fi_path)
                plt.close(fi_fig)
        
        if 'df' in kwargs:
            # Transaction Patterns
            tp_fig = self.plot_transaction_patterns(kwargs['df'])
            tp_path = os.path.join(output_dir, 'transaction_patterns.png')
            tp_fig.savefig(tp_path, dpi=300, bbox_inches='tight')
            plots_saved.append(tp_path)
            plt.close(tp_fig)
        
        if 'validation_scores' in kwargs:
            # Model Comparison
            mc_fig = self.plot_model_comparison(kwargs['validation_scores'])
            if mc_fig:
                mc_path = os.path.join(output_dir, 'model_comparison.png')
                mc_fig.savefig(mc_path, dpi=300, bbox_inches='tight')
                plots_saved.append(mc_path)
                plt.close(mc_fig)
        
        print(f"Saved {len(plots_saved)} plots to {output_dir}")
        return plots_saved
#!/usr/bin/env python3
"""
Main Statistical Analysis Runner for DBPs Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def run_statistical_analysis():
    """Run comprehensive statistical analysis"""
    
    print("Loading dataset...")
    df = pd.read_csv('../data/data.csv')
    if 'Sample' in df.columns:
        df = df.drop('Sample', axis=1)
    
    target_variables = ['DCAA (mg/L)', 'TCAA (ug/L)', 'BCAA(mg/L)', 'HAA5 (ug/L)', 'HAA9 (ug/L)']
    input_features = [col for col in df.columns if col not in target_variables]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Input features: {len(input_features)}")
    print(f"Target variables: {len(target_variables)}")
    
    # 1. Basic descriptive statistics
    print("\n1. Generating descriptive statistics...")
    desc_stats = df.describe()
    desc_stats.to_csv('descriptive_statistics.csv')
    
    # 2. Correlation analysis
    print("2. Performing correlation analysis...")
    corr_matrix = df.corr()
    corr_matrix.to_csv('correlation_matrix.csv')
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution analysis
    print("3. Creating distribution plots...")
    
    # Target distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target in enumerate(target_variables):
        axes[i].hist(df[target], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution: {target}', fontweight='bold')
        axes[i].set_xlabel(target)
        axes[i].set_ylabel('Frequency')
        
        mean_val = df[target].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        axes[i].legend()
    
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('target_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    run_statistical_analysis()
1
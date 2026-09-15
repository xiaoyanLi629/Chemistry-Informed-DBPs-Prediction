#!/usr/bin/env python3
"""
Comprehensive Visualization Suite for DBPs Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

def create_comprehensive_visualizations():
    """Create comprehensive visualizations for the dataset"""
    
    print("Creating comprehensive visualizations...")
    
    # Load data
    df = pd.read_csv('../data/data.csv')
    if 'Sample' in df.columns:
        df = df.drop('Sample', axis=1)
    
    target_variables = ['DCAA (mg/L)', 'TCAA (ug/L)', 'BCAA(mg/L)', 'HAA5 (ug/L)', 'HAA9 (ug/L)']
    input_features = [col for col in df.columns if col not in target_variables]
    
    # 1. Create pairplot for target variables
    print("1. Creating target variables pairplot...")
    plt.figure(figsize=(16, 16))
    
    # Custom pairplot for targets
    n_targets = len(target_variables)
    fig, axes = plt.subplots(n_targets, n_targets, figsize=(16, 16))
    
    for i in range(n_targets):
        for j in range(n_targets):
            if i == j:
                # Diagonal: histograms
                axes[i, j].hist(df[target_variables[i]], bins=15, alpha=0.7, color='skyblue')
                axes[i, j].set_title(target_variables[i], fontweight='bold')
            else:
                # Off-diagonal: scatter plots
                axes[i, j].scatter(df[target_variables[j]], df[target_variables[i]], alpha=0.6)
                # Add correlation coefficient
                corr = df[target_variables[j]].corr(df[target_variables[i]])
                axes[i, j].text(0.1, 0.9, f'r = {corr:.3f}', transform=axes[i, j].transAxes,
                               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            if i == n_targets - 1:
                axes[i, j].set_xlabel(target_variables[j])
            if j == 0:
                axes[i, j].set_ylabel(target_variables[i])
    
    plt.suptitle('Target Variables Pairplot', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('target_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature-Target Relationship Grid
    print("2. Creating feature-target relationship grid...")
    
    n_features = len(input_features)
    n_targets = len(target_variables)
    
    fig, axes = plt.subplots(n_targets, n_features, figsize=(4*n_features, 4*n_targets))
    
    for i, target in enumerate(target_variables):
        for j, feature in enumerate(input_features):
            # Scatter plot with trend line
            axes[i, j].scatter(df[feature], df[target], alpha=0.6, color='steelblue')
            
            # Add trend line
            z = np.polyfit(df[feature], df[target], 1)
            p = np.poly1d(z)
            axes[i, j].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
            
            # Add correlation
            corr = df[feature].corr(df[target])
            axes[i, j].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i, j].transAxes,
                           bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
            
            axes[i, j].set_xlabel(feature)
            axes[i, j].set_ylabel(target)
            axes[i, j].set_title(f'{feature} vs {target}', fontsize=10)
    
    plt.suptitle('Feature-Target Relationships', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('feature_target_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution comparison plots
    print("3. Creating distribution comparison plots...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Target distributions
    from scipy.stats import gaussian_kde
    
    for i, target in enumerate(target_variables):
        if i < 4:  # Only show first 4 targets in first row
            ax = fig.add_subplot(gs[0, i])
            
            # Histogram with KDE
            ax.hist(df[target], bins=15, alpha=0.5, density=True, color='skyblue', label='Histogram')
            
            # KDE curve
            kde = gaussian_kde(df[target])
            x_range = np.linspace(df[target].min(), df[target].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax.set_title(f'Target: {target}', fontweight='bold', fontsize=10)
            ax.set_xlabel('')  # Remove x-label to avoid duplication with title
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=8)
        elif i == 4:  # 5th target variable in second row, first column
            ax = fig.add_subplot(gs[1, 0])
            
            # Histogram with KDE
            ax.hist(df[target], bins=15, alpha=0.5, density=True, color='skyblue', label='Histogram')
            
            # KDE curve
            kde = gaussian_kde(df[target])
            x_range = np.linspace(df[target].min(), df[target].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax.set_title(f'Target: {target}', fontweight='bold', fontsize=10)
            ax.set_xlabel('')  # Remove x-label to avoid duplication with title
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=8)
    
    # Input feature distributions (subset)
    selected_features = input_features[:7]  # Show 7 features to avoid overcrowding
    for i, feature in enumerate(selected_features):
        if i < 3:  # First 3 features in row 1, columns 1-3 (after 5th target)
            col = i + 1
            ax = fig.add_subplot(gs[1, col])
        else:  # Next 4 features in row 2
            col = i - 3
            ax = fig.add_subplot(gs[2, col])
        
        # Histogram with KDE
        ax.hist(df[feature], bins=15, alpha=0.5, density=True, color='lightgreen', label='Histogram')
        
        # KDE curve
        kde = gaussian_kde(df[feature])
        x_range = np.linspace(df[feature].min(), df[feature].max(), 100)
        ax.plot(x_range, kde(x_range), 'orange', linewidth=2, label='KDE')
        
        ax.set_title(f'Feature: {feature}', fontweight='bold', fontsize=10)
        ax.set_xlabel('')  # Remove x-label to avoid duplication with title
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Distribution Analysis with KDE (Targets and Features)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation network visualization
    print("4. Creating correlation network visualization...")
    
    # Create comprehensive correlation matrix
    corr_matrix = df.corr()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Full correlation matrix
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, ax=axes[0,0])
    axes[0,0].set_title('Complete Correlation Matrix', fontweight='bold')
    
    # Feature-target correlations only
    ft_corr = corr_matrix.loc[input_features, target_variables]
    sns.heatmap(ft_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0,1], fmt='.2f')
    axes[0,1].set_title('Feature-Target Correlations', fontweight='bold')
    
    # Target-target correlations
    target_corr = corr_matrix.loc[target_variables, target_variables]
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[1,0], fmt='.3f')
    axes[1,0].set_title('Inter-Target Correlations', fontweight='bold')
    
    # Strong correlations only (absolute > 0.5)
    strong_corr = corr_matrix.where(np.abs(corr_matrix) > 0.5)
    sns.heatmap(strong_corr, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[1,1], fmt='.2f')
    axes[1,1].set_title('Strong Correlations (|r| > 0.5)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('correlation_analysis_suite.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Data quality visualization
    print("5. Creating data quality visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Missing values (should be none, but good to visualize)
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        axes[0,0].text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', 
                      fontsize=16, transform=axes[0,0].transAxes)
        axes[0,0].set_title('Missing Values Analysis', fontweight='bold')
    
    # Data ranges
    data_ranges = df.max() - df.min()
    bars1 = axes[0,1].bar(range(len(data_ranges)), data_ranges)
    axes[0,1].set_xticks(range(len(data_ranges)))
    axes[0,1].set_xticklabels(data_ranges.index, rotation=45)
    axes[0,1].set_title('Data Ranges by Variable', fontweight='bold')
    axes[0,1].set_ylabel('Range (Max - Min)')
    
    # Color bars by variable type
    for i, bar in enumerate(bars1):
        if data_ranges.index[i] in target_variables:
            bar.set_color('lightcoral')
        else:
            bar.set_color('lightblue')
    
    # Coefficient of variation
    cv_values = (df.std() / df.mean()) * 100
    bars2 = axes[1,0].bar(range(len(cv_values)), cv_values)
    axes[1,0].set_xticks(range(len(cv_values)))
    axes[1,0].set_xticklabels(cv_values.index, rotation=45)
    axes[1,0].set_title('Coefficient of Variation (%)', fontweight='bold')
    axes[1,0].set_ylabel('CV (%)')
    
    # Color bars by variable type
    for i, bar in enumerate(bars2):
        if cv_values.index[i] in target_variables:
            bar.set_color('lightcoral')
        else:
            bar.set_color('lightblue')
    
    # Skewness
    skewness = df.skew()
    bars3 = axes[1,1].bar(range(len(skewness)), skewness)
    axes[1,1].set_xticks(range(len(skewness)))
    axes[1,1].set_xticklabels(skewness.index, rotation=45)
    axes[1,1].set_title('Skewness by Variable', fontweight='bold')
    axes[1,1].set_ylabel('Skewness')
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Color bars by skewness magnitude
    for i, bar in enumerate(bars3):
        if abs(skewness.iloc[i]) > 1:
            bar.set_color('red')
        elif abs(skewness.iloc[i]) > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('lightgreen')
    
    plt.tight_layout()
    plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Box plot comparison
    print("6. Creating comprehensive box plot analysis...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    
    # All variables together (normalized)
    df_normalized = (df - df.mean()) / df.std()
    df_normalized.boxplot(ax=axes[0])
    axes[0].set_title('Normalized Variables Box Plot Comparison', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('Standardized Value')
    
    # Input features only
    df[input_features].boxplot(ax=axes[1])
    axes[1].set_title('Input Features Box Plots', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Original Scale')
    
    # Target variables only
    df[target_variables].boxplot(ax=axes[2])
    axes[2].set_title('Target Variables Box Plots', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylabel('Original Scale')
    
    plt.tight_layout()
    plt.savefig('comprehensive_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive visualizations completed!")
    
    # Create visualization summary
    with open('VISUALIZATION_SUMMARY.md', 'w') as f:
        f.write("""# Visualization Analysis Summary

## Generated Visualizations

1. **target_pairplot.png** - Pairwise relationships between all target variables
2. **feature_target_relationships.png** - Scatter plots with trend lines for all feature-target pairs
3. **distribution_comparison.png** - Distribution analysis with histograms and KDE curves
4. **correlation_analysis_suite.png** - Comprehensive correlation analysis including:
   - Complete correlation matrix
   - Feature-target correlations
   - Inter-target correlations
   - Strong correlations only
5. **data_quality_analysis.png** - Data quality assessment including:
   - Missing values analysis
   - Data ranges
   - Coefficient of variation
   - Skewness analysis
6. **comprehensive_boxplots.png** - Box plot comparisons for outlier detection

## Key Insights from Visualizations

### Target Variables
- Pairplot shows relationships and distributions of all DBPs
- Inter-correlations reveal which DBPs tend to vary together

### Feature-Target Relationships
- Correlation patterns help identify most predictive features
- Trend lines show linear relationship strength

### Data Quality
- Distribution shapes indicate need for transformations
- Box plots highlight potential outliers
- Skewness analysis guides preprocessing decisions

## Usage Recommendations

1. Use pairplot to understand target variable relationships
2. Refer to feature-target grid for feature selection
3. Check distribution plots before model preprocessing
4. Use correlation heatmaps for multicollinearity assessment
5. Review box plots for outlier identification and treatment
""")
    
    print("\nVisualization suite completed!")
    print("Generated 6 comprehensive visualization files and 1 summary report.")

if __name__ == "__main__":
    create_comprehensive_visualizations()

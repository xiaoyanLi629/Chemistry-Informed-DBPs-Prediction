#!/usr/bin/env python3
"""
Advanced Statistical Analysis for DBPs Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

def advanced_statistical_analysis():
    """Perform advanced statistical analysis"""
    
    print("Starting advanced statistical analysis...")
    
    # Load data
    df = pd.read_csv('../data/data.csv')
    if 'Sample' in df.columns:
        df = df.drop('Sample', axis=1)
    
    target_variables = ['DCAA (mg/L)', 'TCAA (ug/L)', 'BCAA(mg/L)', 'HAA5 (ug/L)', 'HAA9 (ug/L)']
    input_features = [col for col in df.columns if col not in target_variables]
    
    # 1. Normality Tests
    print("1. Performing normality tests...")
    normality_results = []
    
    for column in df.columns:
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = shapiro(df[column])
        
        # D'Agostino test
        dagostino_stat, dagostino_p = normaltest(df[column])
        
        normality_results.append({
            'Variable': column,
            'Shapiro_Statistic': shapiro_stat,
            'Shapiro_p_value': shapiro_p,
            'DAgostino_Statistic': dagostino_stat,
            'DAgostino_p_value': dagostino_p,
            'Is_Normal_Shapiro': shapiro_p > 0.05,
            'Is_Normal_DAgostino': dagostino_p > 0.05
        })
    
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv('normality_tests.csv', index=False)
    
    # 2. Correlation Analysis with p-values
    print("2. Detailed correlation analysis...")
    correlation_results = []
    
    for feature in input_features:
        for target in target_variables:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(df[feature], df[target])
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(df[feature], df[target])
            
            correlation_results.append({
                'Feature': feature,
                'Target': target,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Pearson_Significant': pearson_p < 0.05,
                'Spearman_Significant': spearman_p < 0.05
            })
    
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df.to_csv('detailed_correlations.csv', index=False)
    
    # 3. Outlier Detection
    print("3. Outlier detection analysis...")
    outlier_results = []
    
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # IQR method
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df[column]))
        zscore_outliers = (z_scores > 3).sum()
        
        outlier_results.append({
            'Variable': column,
            'IQR_Outliers': iqr_outliers,
            'IQR_Outlier_Percentage': (iqr_outliers / len(df)) * 100,
            'ZScore_Outliers': zscore_outliers,
            'ZScore_Outlier_Percentage': (zscore_outliers / len(df)) * 100,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
    
    outlier_df = pd.DataFrame(outlier_results)
    outlier_df.to_csv('outlier_analysis.csv', index=False)
    
    # 4. PCA Analysis
    print("4. Principal Component Analysis...")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[input_features])
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA results
    pca_results = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance_Ratio': np.cumsum(pca.explained_variance_ratio_),
        'Eigenvalue': pca.explained_variance_
    })
    
    pca_results.to_csv('pca_analysis.csv', index=False)
    
    # Create PCA visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Explained variance plot
    axes[0,0].bar(pca_results['PC'], pca_results['Explained_Variance_Ratio'])
    axes[0,0].set_title('Explained Variance by Principal Component', fontweight='bold')
    axes[0,0].set_ylabel('Explained Variance Ratio')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Cumulative variance plot
    axes[0,1].plot(pca_results['PC'], pca_results['Cumulative_Variance_Ratio'], 'bo-')
    axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
    axes[0,1].axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
    axes[0,1].set_title('Cumulative Explained Variance', fontweight='bold')
    axes[0,1].set_ylabel('Cumulative Variance Ratio')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].legend()
    
    # PC1 vs PC2 scatter plot
    axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    axes[1,0].set_title('PC1 vs PC2', fontweight='bold')
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Feature loadings for first 2 PCs
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=input_features
    )
    
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Loadings (PC1 & PC2)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pca_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save loadings
    loadings.to_csv('pca_loadings.csv')
    
    # 5. Variable Summary Statistics
    print("5. Creating comprehensive variable summaries...")
    
    # Detailed statistics for each variable type
    input_detailed = pd.DataFrame({
        'Variable': input_features,
        'Mean': [df[var].mean() for var in input_features],
        'Median': [df[var].median() for var in input_features],
        'Mode': [df[var].mode().iloc[0] if not df[var].mode().empty else np.nan for var in input_features],
        'Std': [df[var].std() for var in input_features],
        'Variance': [df[var].var() for var in input_features],
        'Min': [df[var].min() for var in input_features],
        'Max': [df[var].max() for var in input_features],
        'Range': [df[var].max() - df[var].min() for var in input_features],
        'IQR': [df[var].quantile(0.75) - df[var].quantile(0.25) for var in input_features],
        'CV': [df[var].std() / df[var].mean() for var in input_features],
        'Skewness': [df[var].skew() for var in input_features],
        'Kurtosis': [df[var].kurtosis() for var in input_features]
    })
    
    target_detailed = pd.DataFrame({
        'Variable': target_variables,
        'Mean': [df[var].mean() for var in target_variables],
        'Median': [df[var].median() for var in target_variables],
        'Mode': [df[var].mode().iloc[0] if not df[var].mode().empty else np.nan for var in target_variables],
        'Std': [df[var].std() for var in target_variables],
        'Variance': [df[var].var() for var in target_variables],
        'Min': [df[var].min() for var in target_variables],
        'Max': [df[var].max() for var in target_variables],
        'Range': [df[var].max() - df[var].min() for var in target_variables],
        'IQR': [df[var].quantile(0.75) - df[var].quantile(0.25) for var in target_variables],
        'CV': [df[var].std() / df[var].mean() for var in target_variables],
        'Skewness': [df[var].skew() for var in target_variables],
        'Kurtosis': [df[var].kurtosis() for var in target_variables]
    })
    
    input_detailed.to_csv('input_detailed_statistics.csv', index=False)
    target_detailed.to_csv('target_detailed_statistics.csv', index=False)
    
    print("Advanced statistical analysis completed!")
    
    # Create summary report
    with open('ADVANCED_ANALYSIS_SUMMARY.md', 'w') as f:
        f.write(f"""# Advanced Statistical Analysis Summary

## Analysis Overview
- Dataset size: {df.shape[0]} samples, {df.shape[1]} variables
- Input features: {len(input_features)}
- Target variables: {len(target_variables)}

## Key Findings

### Normality Tests
- Variables following normal distribution (Shapiro-Wilk): {sum(normality_df['Is_Normal_Shapiro'])}
- Variables following normal distribution (D'Agostino): {sum(normality_df['Is_Normal_DAgostino'])}

### Outlier Analysis
- Average outliers per variable (IQR method): {outlier_df['IQR_Outliers'].mean():.1f}
- Average outliers per variable (Z-score method): {outlier_df['ZScore_Outliers'].mean():.1f}

### PCA Analysis
- Components for 80% variance: {(pca_results['Cumulative_Variance_Ratio'] >= 0.8).argmax() + 1}
- Components for 95% variance: {(pca_results['Cumulative_Variance_Ratio'] >= 0.95).argmax() + 1}

### Correlation Analysis
- Significant Pearson correlations (p<0.05): {sum(correlation_df['Pearson_Significant'])}
- Significant Spearman correlations (p<0.05): {sum(correlation_df['Spearman_Significant'])}

## Generated Files
1. normality_tests.csv - Normality test results for all variables
2. detailed_correlations.csv - Correlation analysis with p-values
3. outlier_analysis.csv - Comprehensive outlier detection results
4. pca_analysis.csv - Principal component analysis results
5. pca_loadings.csv - Feature loadings for principal components
6. input_detailed_statistics.csv - Detailed statistics for input features
7. target_detailed_statistics.csv - Detailed statistics for target variables
8. pca_analysis_plots.png - PCA visualization plots

## Recommendations
1. Consider transformations for non-normal variables
2. Investigate outliers identified by both methods
3. Use PCA results for dimensionality reduction if needed
4. Focus on significant correlations for feature selection
""")
    
    print("\nGenerated files:")
    print("- 7 CSV files with detailed analysis results")
    print("- 1 PNG file with PCA visualizations")
    print("- 1 Advanced analysis summary report")

if __name__ == "__main__":
    advanced_statistical_analysis()

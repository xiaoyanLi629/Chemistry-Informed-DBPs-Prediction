#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Statistical Analysis for DBPs Prediction Project
Author: Statistical Analysis Module
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, normaltest, kstest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class DBPsStatisticalAnalysis:
    def __init__(self, data_path='../data/data.csv'):
        """Initialize the statistical analysis class"""
        self.data_path = data_path
        self.df = None
        self.input_features = None
        self.target_variables = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Remove Sample column as it's just an identifier
        if 'Sample' in self.df.columns:
            self.df = self.df.drop('Sample', axis=1)
        
        # Define input features and target variables based on the project structure
        self.target_variables = ['DCAA (mg/L)', 'TCAA (ug/L)', 'BCAA(mg/L)', 'HAA5 (ug/L)', 'HAA9 (ug/L)']
        self.input_features = [col for col in self.df.columns if col not in self.target_variables]
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Input features: {len(self.input_features)}")
        print(f"Target variables: {len(self.target_variables)}")
        
    def basic_statistics(self):
        """Generate basic descriptive statistics"""
        print("\n=== BASIC DESCRIPTIVE STATISTICS ===")
        
        # Overall statistics
        desc_stats = self.df.describe()
        
        # Additional statistics
        additional_stats = pd.DataFrame({
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Skewness': self.df.skew(),
            'Kurtosis': self.df.kurtosis()
        })
        
        # Combine statistics
        full_stats = pd.concat([desc_stats.T, additional_stats], axis=1)
        
        # Save results
        full_stats.to_csv('analysis/eda/descriptive_statistics.csv')
        self.results['descriptive_stats'] = full_stats
        
        # Separate statistics for inputs and targets
        input_stats = full_stats.loc[self.input_features]
        target_stats = full_stats.loc[self.target_variables]
        
        input_stats.to_csv('analysis/eda/input_features_statistics.csv')
        target_stats.to_csv('analysis/eda/target_variables_statistics.csv')
        
        print("Basic statistics saved to CSV files.")
        return full_stats
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Pearson correlation
        pearson_corr = self.df.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = self.df.corr(method='spearman')
        
        # Save correlation matrices
        pearson_corr.to_csv('analysis/eda/pearson_correlation_matrix.csv')
        spearman_corr.to_csv('analysis/eda/spearman_correlation_matrix.csv')
        
        # Create correlation heatmaps
        fig, axes = plt.subplots(2, 1, figsize=(16, 20))
        
        # Pearson correlation heatmap
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], fmt='.3f')
        axes[0].set_title('Pearson Correlation Matrix', fontsize=16, fontweight='bold')
        
        # Spearman correlation heatmap
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[1], fmt='.3f')
        axes[1].set_title('Spearman Correlation Matrix', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature-target correlations
        feature_target_corr = pd.DataFrame()
        for target in self.target_variables:
            corr_data = []
            for feature in self.input_features:
                pearson_r, pearson_p = pearsonr(self.df[feature], self.df[target])
                spearman_r, spearman_p = spearmanr(self.df[feature], self.df[target])
                corr_data.append({
                    'Feature': feature,
                    'Target': target,
                    'Pearson_r': pearson_r,
                    'Pearson_p': pearson_p,
                    'Spearman_r': spearman_r,
                    'Spearman_p': spearman_p
                })
            temp_df = pd.DataFrame(corr_data)
            feature_target_corr = pd.concat([feature_target_corr, temp_df], ignore_index=True)
        
        feature_target_corr.to_csv('analysis/eda/feature_target_correlations.csv', index=False)
        
        self.results['pearson_corr'] = pearson_corr
        self.results['spearman_corr'] = spearman_corr
        self.results['feature_target_corr'] = feature_target_corr
        
        print("Correlation analysis completed and saved.")
        
    def distribution_analysis(self):
        """Analyze distributions of variables"""
        print("\n=== DISTRIBUTION ANALYSIS ===")
        
        # Create distribution plots
        n_vars = len(self.df.columns)
        n_cols = 4
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        normality_results = []
        
        for i, column in enumerate(self.df.columns):
            # Distribution plot
            axes[i].hist(self.df[column], bins=15, alpha=0.7, density=True, color='skyblue')
            axes[i].set_title(f'Distribution: {column}', fontweight='bold')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            
            # Add mean and median lines
            mean_val = self.df[column].mean()
            median_val = self.df[column].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
            axes[i].legend()
            
            # Normality tests
            shapiro_stat, shapiro_p = shapiro(self.df[column])
            dagostino_stat, dagostino_p = normaltest(self.df[column])
            
            normality_results.append({
                'Variable': column,
                'Shapiro_Statistic': shapiro_stat,
                'Shapiro_p_value': shapiro_p,
                'DAgostino_Statistic': dagostino_stat,
                'DAgostino_p_value': dagostino_p,
                'Is_Normal_Shapiro': shapiro_p > 0.05,
                'Is_Normal_DAgostino': dagostino_p > 0.05
            })
        
        # Remove empty subplots
        for i in range(len(self.df.columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('variable_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save normality test results
        normality_df = pd.DataFrame(normality_results)
        normality_df.to_csv('analysis/eda/normality_tests.csv', index=False)
        
        self.results['normality_tests'] = normality_df
        
        print("Distribution analysis completed and saved.")
    
    def feature_importance_analysis(self):
        """Analyze feature importance using mutual information"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        X = self.df[self.input_features]
        importance_results = []
        
        for target in self.target_variables:
            y = self.df[target]
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            for i, feature in enumerate(self.input_features):
                importance_results.append({
                    'Feature': feature,
                    'Target': target,
                    'Mutual_Information': mi_scores[i]
                })
        
        importance_df = pd.DataFrame(importance_results)
        importance_df.to_csv('analysis/eda/feature_importance_mutual_info.csv', index=False)
        
        # Create feature importance visualization
        fig, axes = plt.subplots(len(self.target_variables), 1, figsize=(12, 4*len(self.target_variables)))
        if len(self.target_variables) == 1:
            axes = [axes]
        
        for i, target in enumerate(self.target_variables):
            target_importance = importance_df[importance_df['Target'] == target].sort_values('Mutual_Information', ascending=True)
            
            axes[i].barh(target_importance['Feature'], target_importance['Mutual_Information'])
            axes[i].set_title(f'Feature Importance for {target}', fontweight='bold')
            axes[i].set_xlabel('Mutual Information Score')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['feature_importance'] = importance_df
        
        print("Feature importance analysis completed and saved.")
    
    def pca_analysis(self):
        """Perform Principal Component Analysis"""
        print("\n=== PRINCIPAL COMPONENT ANALYSIS ===")
        
        # Standardize the input features
        X = self.df[self.input_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA results dataframe
        pca_results = pd.DataFrame({
            'Principal_Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Explained_Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Variance_Ratio': np.cumsum(pca.explained_variance_ratio_),
            'Eigenvalue': pca.explained_variance_
        })
        
        pca_results.to_csv('analysis/eda/pca_results.csv', index=False)
        
        # Create PCA visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Explained variance plot
        axes[0,0].bar(pca_results['Principal_Component'], pca_results['Explained_Variance_Ratio'])
        axes[0,0].set_title('Explained Variance Ratio by Principal Component', fontweight='bold')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Cumulative variance plot
        axes[0,1].plot(pca_results['Principal_Component'], pca_results['Cumulative_Variance_Ratio'], 'bo-')
        axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
        axes[0,1].axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
        axes[0,1].set_title('Cumulative Explained Variance', fontweight='bold')
        axes[0,1].set_ylabel('Cumulative Variance Ratio')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        
        # PC1 vs PC2 scatter plot
        axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        axes[1,0].set_title('PC1 vs PC2 Scatter Plot', fontweight='bold')
        axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Feature loadings heatmap
        loadings = pd.DataFrame(
            pca.components_[:4].T,
            columns=[f'PC{i+1}' for i in range(4)],
            index=self.input_features
        )
        
        sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Feature Loadings (First 4 PCs)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save loadings
        loadings.to_csv('analysis/eda/pca_loadings.csv')
        
        self.results['pca'] = pca_results
        self.results['pca_loadings'] = loadings
        
        print("PCA analysis completed and saved.")
    
    def target_variable_analysis(self):
        """Detailed analysis of target variables"""
        print("\n=== TARGET VARIABLE ANALYSIS ===")
        
        # Inter-target correlations
        target_corr = self.df[self.target_variables].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f')
        plt.title('Inter-Target Variable Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        target_corr.to_csv('analysis/eda/target_correlation_matrix.csv')
        
        # Target variable distributions comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, target in enumerate(self.target_variables):
            # Box plot
            axes[i].boxplot(self.df[target])
            axes[i].set_title(f'Box Plot: {target}', fontweight='bold')
            axes[i].set_ylabel('Value')
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('target_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Target statistics summary
        target_summary = pd.DataFrame({
            'Variable': self.target_variables,
            'Mean': [self.df[var].mean() for var in self.target_variables],
            'Median': [self.df[var].median() for var in self.target_variables],
            'Std': [self.df[var].std() for var in self.target_variables],
            'Min': [self.df[var].min() for var in self.target_variables],
            'Max': [self.df[var].max() for var in self.target_variables],
            'Range': [self.df[var].max() - self.df[var].min() for var in self.target_variables],
            'CV': [self.df[var].std() / self.df[var].mean() for var in self.target_variables]
        })
        
        target_summary.to_csv('analysis/eda/target_summary_statistics.csv', index=False)
        
        self.results['target_correlations'] = target_corr
        self.results['target_summary'] = target_summary
        
        print("Target variable analysis completed and saved.")
    
    def outlier_analysis(self):
        """Detect and analyze outliers"""
        print("\n=== OUTLIER ANALYSIS ===")
        
        outlier_results = []
        
        for column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # IQR method
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)].index.tolist()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[column]))
            zscore_outliers = self.df[z_scores > 3].index.tolist()
            
            outlier_results.append({
                'Variable': column,
                'IQR_Outlier_Count': len(iqr_outliers),
                'IQR_Outlier_Percentage': (len(iqr_outliers) / len(self.df)) * 100,
                'ZScore_Outlier_Count': len(zscore_outliers),
                'ZScore_Outlier_Percentage': (len(zscore_outliers) / len(self.df)) * 100,
                'IQR_Lower_Bound': lower_bound,
                'IQR_Upper_Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_results)
        outlier_df.to_csv('analysis/eda/outlier_analysis.csv', index=False)
        
        self.results['outliers'] = outlier_df
        
        print("Outlier analysis completed and saved.")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n=== GENERATING SUMMARY REPORT ===")
        
        report = []
        report.append("# DBPs Dataset Statistical Analysis Summary Report\n")
        report.append(f"Dataset Shape: {self.df.shape}\n")
        report.append(f"Number of Input Features: {len(self.input_features)}\n")
        report.append(f"Number of Target Variables: {len(self.target_variables)}\n")
        report.append(f"Total Missing Values: {self.df.isnull().sum().sum()}\n\n")
        
        # Key findings
        report.append("## Key Statistical Findings\n\n")
        
        # Most correlated feature-target pairs
        if 'feature_target_corr' in self.results:
            top_correlations = self.results['feature_target_corr'].groupby('Target').apply(
                lambda x: x.loc[x['Pearson_r'].abs().idxmax()]
            )
            report.append("### Strongest Feature-Target Correlations:\n")
            for idx, row in top_correlations.iterrows():
                report.append(f"- {row['Target']}: {row['Feature']} (r = {row['Pearson_r']:.3f})\n")
        
        # Normality test summary
        if 'normality_tests' in self.results:
            normal_vars = self.results['normality_tests'][
                (self.results['normality_tests']['Is_Normal_Shapiro']) & 
                (self.results['normality_tests']['Is_Normal_DAgostino'])
            ]['Variable'].tolist()
            report.append(f"\n### Variables Following Normal Distribution: {len(normal_vars)}/{len(self.df.columns)}\n")
            for var in normal_vars:
                report.append(f"- {var}\n")
        
        # PCA summary
        if 'pca' in self.results:
            pca_80 = (self.results['pca']['Cumulative_Variance_Ratio'] >= 0.8).idxmax() + 1
            pca_95 = (self.results['pca']['Cumulative_Variance_Ratio'] >= 0.95).idxmax() + 1
            report.append(f"\n### PCA Summary:\n")
            report.append(f"- Components for 80% variance: {pca_80}\n")
            report.append(f"- Components for 95% variance: {pca_95}\n")
        
        # Write report
        with open('analysis/eda/statistical_analysis_report.md', 'w') as f:
            f.writelines(report)
        
        print("Summary report generated and saved.")
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis pipeline"""
        print("Starting comprehensive statistical analysis...")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.basic_statistics()
        self.correlation_analysis()
        self.distribution_analysis()
        self.feature_importance_analysis()
        self.pca_analysis()
        self.target_variable_analysis()
        self.outlier_analysis()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
        print("All results saved in the 'statistics' folder.")
        print("="*60)

if __name__ == "__main__":
    # Create and run the analysis
    analyzer = DBPsStatisticalAnalysis()
    analyzer.run_complete_analysis() 
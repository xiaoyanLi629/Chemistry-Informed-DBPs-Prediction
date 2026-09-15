#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visualization Analysis for DBPs Dataset
Author: Visualization Analysis Module
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class DBPsVisualizationAnalysis:
    def __init__(self, data_path='../data/data.csv'):
        """Initialize the visualization analysis class"""
        self.data_path = data_path
        self.df = None
        self.input_features = None
        self.target_variables = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset for visualization analysis...")
        self.df = pd.read_csv(self.data_path)
        
        # Remove Sample column
        if 'Sample' in self.df.columns:
            self.df = self.df.drop('Sample', axis=1)
        
        # Define variables
        self.target_variables = ['DCAA (mg/L)', 'TCAA (ug/L)', 'BCAA(mg/L)', 'HAA5 (ug/L)', 'HAA9 (ug/L)']
        self.input_features = [col for col in self.df.columns if col not in self.target_variables]
        
        print(f"Data loaded successfully! Shape: {self.df.shape}")
    
    def create_pairplot_analysis(self):
        """Create comprehensive pairplot analysis"""
        print("Creating pairplot analysis...")
        
        # Target variables pairplot
        plt.figure(figsize=(16, 16))
        sns.pairplot(self.df[self.target_variables], diag_kind='hist', corner=True)
        plt.suptitle('Target Variables Pairplot Analysis', y=1.02, fontsize=16, fontweight='bold')
        plt.savefig('target_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Input features pairplot (subset due to size)
        if len(self.input_features) > 6:
            # Select most important features for visualization
            selected_features = self.input_features[:6]
        else:
            selected_features = self.input_features
            
        plt.figure(figsize=(16, 16))
        sns.pairplot(self.df[selected_features], diag_kind='hist', corner=True)
        plt.suptitle('Input Features Pairplot Analysis', y=1.02, fontsize=16, fontweight='bold')
        plt.savefig('input_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Pairplot analysis completed.")
    
    def create_feature_target_relationships(self):
        """Create detailed feature-target relationship plots"""
        print("Creating feature-target relationship analysis...")
        
        n_targets = len(self.target_variables)
        n_features = len(self.input_features)
        
        # Create scatter plots for each feature vs each target
        fig, axes = plt.subplots(n_targets, n_features, figsize=(4*n_features, 4*n_targets))
        
        for i, target in enumerate(self.target_variables):
            for j, feature in enumerate(self.input_features):
                # Calculate correlation
                corr = np.corrcoef(self.df[feature], self.df[target])[0, 1]
                
                # Create scatter plot
                axes[i, j].scatter(self.df[feature], self.df[target], alpha=0.6)
                axes[i, j].set_xlabel(feature)
                axes[i, j].set_ylabel(target)
                axes[i, j].set_title(f'r = {corr:.3f}', fontweight='bold')
                
                # Add trend line
                z = np.polyfit(self.df[feature], self.df[target], 1)
                p = np.poly1d(z)
                axes[i, j].plot(self.df[feature], p(self.df[feature]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('feature_target_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature-target relationship analysis completed.")
    
    def create_distribution_comparison(self):
        """Create distribution comparison plots"""
        print("Creating distribution comparison analysis...")
        
        # Import scipy.stats for KDE
        from scipy import stats
        
        # Combined distribution plot
        n_vars = len(self.df.columns)
        n_cols = 4
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, column in enumerate(self.df.columns):
            # Histogram with KDE
            axes[i].hist(self.df[column], bins=15, alpha=0.5, density=True, label='Histogram')
            x_range = np.linspace(self.df[column].min(), self.df[column].max(), 100)
            kde_values = stats.gaussian_kde(self.df[column])(x_range)
            axes[i].plot(x_range, kde_values, 'r-', label='KDE')
            axes[i].set_title(f'{column}', fontweight='bold')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            axes[i].legend(loc='upper right', fontsize=8)
        
        # Remove empty subplots
        for i in range(len(self.df.columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Box plots comparison
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Input features boxplots
        self.df[self.input_features].boxplot(ax=axes[0])
        axes[0].set_title('Input Features Distribution (Box Plots)', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Target variables boxplots
        self.df[self.target_variables].boxplot(ax=axes[1])
        axes[1].set_title('Target Variables Distribution (Box Plots)', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Distribution comparison analysis completed.")
    
    def create_correlation_analysis_detailed(self):
        """Create detailed correlation analysis visualizations"""
        print("Creating detailed correlation analysis...")
        
        # Create a comprehensive correlation plot
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main correlation heatmap
        ax1 = fig.add_subplot(gs[0, :])
        corr_matrix = self.df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1, fmt='.2f', cbar_kws={"shrink": .8})
        ax1.set_title('Complete Correlation Matrix (Lower Triangle)', fontsize=16, fontweight='bold')
        
        # Feature-target correlation bar plot
        ax2 = fig.add_subplot(gs[1, 0])
        feature_target_corr = []
        for target in self.target_variables:
            for feature in self.input_features:
                corr_val = corr_matrix.loc[feature, target]
                feature_target_corr.append({'Feature': feature, 'Target': target, 'Correlation': corr_val})
        
        ft_corr_df = pd.DataFrame(feature_target_corr)
        avg_corr = ft_corr_df.groupby('Feature')['Correlation'].mean().sort_values(ascending=True)
        
        avg_corr.plot(kind='barh', ax=ax2)
        ax2.set_title('Average Feature-Target Correlations', fontweight='bold')
        ax2.set_xlabel('Average Correlation')
        
        # Target-target correlation
        ax3 = fig.add_subplot(gs[1, 1])
        target_corr = self.df[self.target_variables].corr()
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, square=True, ax=ax3, fmt='.3f')
        ax3.set_title('Target Variables Inter-Correlations', fontweight='bold')
        
        # Feature-feature correlation (top correlations)
        ax4 = fig.add_subplot(gs[2, :])
        feature_corr = self.df[self.input_features].corr()
        
        # Get top correlations (excluding diagonal)
        upper_tri = np.triu(np.ones_like(feature_corr, dtype=bool), k=1)
        feature_corr_upper = feature_corr.where(upper_tri)
        
        # Find top 10 correlations
        feature_pairs = []
        for i in range(len(self.input_features)):
            for j in range(i+1, len(self.input_features)):
                feature_pairs.append({
                    'Pair': f"{self.input_features[i]} - {self.input_features[j]}",
                    'Correlation': feature_corr.iloc[i, j]
                })
        
        feature_pairs_df = pd.DataFrame(feature_pairs)
        top_pairs = feature_pairs_df.nlargest(10, 'Correlation')
        
        bars = ax4.barh(range(len(top_pairs)), top_pairs['Correlation'])
        ax4.set_yticks(range(len(top_pairs)))
        ax4.set_yticklabels(top_pairs['Pair'])
        ax4.set_title('Top 10 Feature-Feature Correlations', fontweight='bold')
        ax4.set_xlabel('Correlation')
        
        # Color bars based on correlation strength
        for i, bar in enumerate(bars):
            if top_pairs.iloc[i]['Correlation'] > 0.7:
                bar.set_color('red')
            elif top_pairs.iloc[i]['Correlation'] > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        plt.savefig('detailed_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Detailed correlation analysis completed.")
    
    def create_data_quality_visualization(self):
        """Create data quality and outlier visualizations"""
        print("Creating data quality visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Missing values heatmap
        missing_data = self.df.isnull()
        if missing_data.sum().sum() > 0:
            sns.heatmap(missing_data, cbar=True, ax=axes[0, 0])
            axes[0, 0].set_title('Missing Values Heatmap', fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=16)
            axes[0, 0].set_title('Missing Values Status', fontweight='bold')
        
        # Outlier detection using IQR method
        outlier_counts = []
        for column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
            outlier_counts.append(outliers)
        
        axes[0, 1].bar(range(len(self.df.columns)), outlier_counts)
        axes[0, 1].set_xticks(range(len(self.df.columns)))
        axes[0, 1].set_xticklabels(self.df.columns, rotation=45)
        axes[0, 1].set_title('Outlier Count by Variable (IQR Method)', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Outliers')
        
        # Data range visualization
        data_ranges = self.df.max() - self.df.min()
        axes[1, 0].bar(range(len(data_ranges)), data_ranges)
        axes[1, 0].set_xticks(range(len(data_ranges)))
        axes[1, 0].set_xticklabels(data_ranges.index, rotation=45)
        axes[1, 0].set_title('Data Range by Variable', fontweight='bold')
        axes[1, 0].set_ylabel('Range (Max - Min)')
        
        # Coefficient of variation
        cv_values = (self.df.std() / self.df.mean()) * 100
        axes[1, 1].bar(range(len(cv_values)), cv_values)
        axes[1, 1].set_xticks(range(len(cv_values)))
        axes[1, 1].set_xticklabels(cv_values.index, rotation=45)
        axes[1, 1].set_title('Coefficient of Variation by Variable', fontweight='bold')
        axes[1, 1].set_ylabel('CV (%)')
        
        plt.tight_layout()
        plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Data quality visualization completed.")
    
    def create_advanced_target_analysis(self):
        """Create advanced target variable analysis"""
        print("Creating advanced target analysis...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Target variable trends (if there's an order to samples)
        axes[0, 0].plot(self.df[self.target_variables].values)
        axes[0, 0].set_title('Target Variables Trends Across Samples', fontweight='bold')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend(self.target_variables, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Target variable ratios
        ratios = pd.DataFrame()
        for i, target1 in enumerate(self.target_variables):
            for j, target2 in enumerate(self.target_variables):
                if i < j:
                    ratio_name = f'{target1}/{target2}'
                    ratios[ratio_name] = self.df[target1] / self.df[target2]
        
        if not ratios.empty:
            ratios.boxplot(ax=axes[0, 1])
            axes[0, 1].set_title('Target Variable Ratios Distribution', fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Target variable clustering/grouping analysis
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        target_scaled = scaler.fit_transform(self.df[self.target_variables])
        
        # Determine optimal number of clusters
        inertias = []
        k_range = range(1, min(8, len(self.df)//2))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(target_scaled)
            inertias.append(kmeans.inertia_)
        
        axes[1, 0].plot(k_range, inertias, 'bo-')
        axes[1, 0].set_title('Elbow Method for Optimal Clusters', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Inertia')
        
        # Apply clustering with optimal k (let's use k=3)
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(target_scaled)
        
        # Scatter plot of first two targets colored by cluster
        scatter = axes[1, 1].scatter(self.df[self.target_variables[0]], self.df[self.target_variables[1]], 
                                   c=clusters, cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel(self.target_variables[0])
        axes[1, 1].set_ylabel(self.target_variables[1])
        axes[1, 1].set_title('Target Variables Clustering', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1])
        
        # Target variable correlations with input features (heatmap)
        feature_target_corr_matrix = np.zeros((len(self.input_features), len(self.target_variables)))
        for i, feature in enumerate(self.input_features):
            for j, target in enumerate(self.target_variables):
                feature_target_corr_matrix[i, j] = np.corrcoef(self.df[feature], self.df[target])[0, 1]
        
        im = axes[2, 0].imshow(feature_target_corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[2, 0].set_xticks(range(len(self.target_variables)))
        axes[2, 0].set_xticklabels(self.target_variables, rotation=45)
        axes[2, 0].set_yticks(range(len(self.input_features)))
        axes[2, 0].set_yticklabels(self.input_features)
        axes[2, 0].set_title('Feature-Target Correlation Matrix', fontweight='bold')
        plt.colorbar(im, ax=axes[2, 0])
        
        # Target variable importance based on variance
        target_variance = self.df[self.target_variables].var()
        axes[2, 1].bar(range(len(target_variance)), target_variance)
        axes[2, 1].set_xticks(range(len(target_variance)))
        axes[2, 1].set_xticklabels(target_variance.index, rotation=45)
        axes[2, 1].set_title('Target Variable Variance', fontweight='bold')
        axes[2, 1].set_ylabel('Variance')
        
        plt.tight_layout()
        plt.savefig('advanced_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Advanced target analysis completed.")
    
    def create_model_input_analysis(self):
        """Create analysis specifically for model input understanding"""
        print("Creating model input analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Feature scaling comparison
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        scalers = {'Original': None, 'StandardScaler': StandardScaler(), 
                  'MinMaxScaler': MinMaxScaler(), 'RobustScaler': RobustScaler()}
        
        # Compare distributions after different scaling methods
        feature_to_analyze = self.input_features[0]  # Analyze first feature as example
        
        for i, (name, scaler) in enumerate(scalers.items()):
            if scaler is None:
                data_to_plot = self.df[feature_to_analyze]
            else:
                data_to_plot = scaler.fit_transform(self.df[[feature_to_analyze]]).flatten()
            
            axes[0, 0].hist(data_to_plot, alpha=0.5, label=name, bins=15)
        
        axes[0, 0].set_title(f'Feature Scaling Comparison: {feature_to_analyze}', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Feature importance based on correlation with all targets
        feature_importance = []
        for feature in self.input_features:
            avg_corr = np.mean([abs(np.corrcoef(self.df[feature], self.df[target])[0, 1]) 
                               for target in self.target_variables])
            feature_importance.append(avg_corr)
        
        sorted_features = sorted(zip(self.input_features, feature_importance), 
                               key=lambda x: x[1], reverse=True)
        
        features, importances = zip(*sorted_features)
        axes[0, 1].barh(range(len(features)), importances)
        axes[0, 1].set_yticks(range(len(features)))
        axes[0, 1].set_yticklabels(features)
        axes[0, 1].set_title('Feature Importance (Avg Correlation with Targets)', fontweight='bold')
        axes[0, 1].set_xlabel('Average Absolute Correlation')
        
        # Input feature multicollinearity check
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.input_features
        vif_data["VIF"] = [variance_inflation_factor(self.df[self.input_features].values, i) 
                          for i in range(len(self.input_features))]
        
        bars = axes[1, 0].bar(range(len(vif_data)), vif_data["VIF"])
        axes[1, 0].set_xticks(range(len(vif_data)))
        axes[1, 0].set_xticklabels(vif_data["Feature"], rotation=45)
        axes[1, 0].set_title('Variance Inflation Factor (Multicollinearity Check)', fontweight='bold')
        axes[1, 0].set_ylabel('VIF')
        axes[1, 0].axhline(y=5, color='r', linestyle='--', label='VIF = 5 (threshold)')
        axes[1, 0].axhline(y=10, color='r', linestyle='-', label='VIF = 10 (high multicollinearity)')
        axes[1, 0].legend()
        
        # Color bars based on VIF values
        for i, bar in enumerate(bars):
            if vif_data.iloc[i]["VIF"] > 10:
                bar.set_color('red')
            elif vif_data.iloc[i]["VIF"] > 5:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        # Data balance and distribution summary
        summary_stats = self.df.describe().T
        summary_stats['CV'] = summary_stats['std'] / summary_stats['mean']
        
        # Plot coefficient of variation
        axes[1, 1].bar(range(len(summary_stats)), summary_stats['CV'])
        axes[1, 1].set_xticks(range(len(summary_stats)))
        axes[1, 1].set_xticklabels(summary_stats.index, rotation=45)
        axes[1, 1].set_title('Coefficient of Variation by Variable', fontweight='bold')
        axes[1, 1].set_ylabel('CV (std/mean)')
        
        plt.tight_layout()
        plt.savefig('model_input_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save VIF analysis
        vif_data.to_csv('analysis/eda/variance_inflation_factors.csv', index=False)
        
        print("Model input analysis completed.")
    
    def run_complete_visualization(self):
        """Run the complete visualization analysis pipeline"""
        print("Starting comprehensive visualization analysis...")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run all visualizations
        self.create_pairplot_analysis()
        self.create_feature_target_relationships()
        self.create_distribution_comparison()
        self.create_correlation_analysis_detailed()
        self.create_data_quality_visualization()
        self.create_advanced_target_analysis()
        self.create_model_input_analysis()
        
        print("\n" + "="*60)
        print("VISUALIZATION ANALYSIS COMPLETED SUCCESSFULLY!")
        print("All visualizations saved in the 'statistics' folder.")
        print("="*60)

if __name__ == "__main__":
    # Create and run the visualization analysis
    visualizer = DBPsVisualizationAnalysis()
    visualizer.run_complete_visualization() 
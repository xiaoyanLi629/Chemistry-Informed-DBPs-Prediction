# Advanced Statistical Analysis Summary

## Analysis Overview
- Dataset size: 66 samples, 13 variables
- Input features: 8
- Target variables: 5

## Key Findings

### Normality Tests
- Variables following normal distribution (Shapiro-Wilk): 2
- Variables following normal distribution (D'Agostino): 1

### Outlier Analysis
- Average outliers per variable (IQR method): 1.5
- Average outliers per variable (Z-score method): 0.5

### PCA Analysis
- Components for 80% variance: 4
- Components for 95% variance: 6

### Correlation Analysis
- Significant Pearson correlations (p<0.05): 12
- Significant Spearman correlations (p<0.05): 17

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

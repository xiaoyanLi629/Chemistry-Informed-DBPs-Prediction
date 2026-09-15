# DBPs统计分析套件

这个目录包含了DBPs（消毒副产品）预测数据集的综合统计分析套件。

## 概述

统计分析套件通过多种分析方法为数据集提供详细的洞察：

- **描述性统计**: 所有变量的基础和高级描述性统计
- **相关性分析**: Pearson和Spearman相关性以及显著性检验
- **分布分析**: 正态性检验和分布可视化
- **主成分分析**: 降维和特征重要性分析
- **异常值检测**: 多种异常值识别方法
- **综合可视化**: 多种图表类型用于数据探索
- **数据质量评估**: 全面的数据质量检查和可视化

## 文件结构与说明

### 核心分析脚本

#### 1. **`run_analysis.py`** - 基础统计分析
**功能**:
- 描述性统计计算
- 相关性矩阵和热图
- 所有变量的分布图
- 异常值检测的箱线图
- 汇总表格生成

**输出文件**:
- `descriptive_statistics.csv` - 基础描述性统计
- `correlation_matrix.csv` - 完整相关性矩阵
- `correlation_heatmap.png` - 相关性热图
- `target_distributions.png` - 目标变量分布图

#### 2. **`comprehensive_analysis.py`** - 综合统计分析
**功能**:
- 完整的描述性统计分析
- 双重相关性分析（Pearson & Spearman）
- 变量分布分析
- 特征重要性分析（互信息）
- 主成分分析（PCA）
- 目标变量专项分析
- 异常值检测（IQR & Z-score方法）
- 自动化汇总报告生成

**输出文件**:
- `descriptive_statistics.csv` - 完整描述性统计
- `input_features_statistics.csv` - 输入特征统计
- `target_variables_statistics.csv` - 目标变量统计
- `pearson_correlation_matrix.csv` - Pearson相关性矩阵
- `spearman_correlation_matrix.csv` - Spearman相关性矩阵
- `feature_target_correlations.csv` - 特征-目标相关性
- `correlation_heatmaps.png` - 双重相关性热图
- `variable_distributions.png` - 变量分布图
- `normality_tests.csv` - 正态性检验结果
- `feature_importance_scores.csv` - 特征重要性评分
- `pca_results.csv` - PCA分析结果
- `pca_loadings.csv` - PCA载荷
- `outlier_detection_results.csv` - 异常值检测结果

#### 3. **`advanced_analysis.py`** - 高级统计分析
**功能**:
- 正态性检验（Shapiro-Wilk, D'Agostino）
- 详细相关性分析（含p值）
- 综合异常值检测（IQR和Z-score方法）
- 主成分分析（PCA）
- 详细变量统计

**输出文件**:
- `normality_tests.csv` - 正态性检验结果
- `detailed_correlations.csv` - 详细相关性分析（含p值）
- `outlier_analysis.csv` - 异常值检测结果
- `pca_analysis.csv` - PCA结果和解释方差
- `pca_loadings.csv` - 主成分特征载荷
- `input_detailed_statistics.csv` - 输入特征详细统计
- `target_detailed_statistics.csv` - 目标变量详细统计
- `pca_analysis_plots.png` - PCA可视化套件

#### 4. **`visualization_analysis.py`** - 高级可视化分析
**功能**:
- 配对图分析（目标变量和输入特征）
- 特征-目标关系网格图
- 分布比较（含KDE密度估计）
- 详细相关性分析可视化
- 数据质量评估图表
- 高级目标变量分析
- 模型输入分析可视化

**输出文件**:
- `target_pairplot.png` - 目标变量配对图
- `input_pairplot.png` - 输入特征配对图
- `feature_target_relationships.png` - 特征-目标关系图
- `distribution_comparison.png` - 分布比较图
- `boxplot_comparison.png` - 箱线图比较
- `correlation_analysis_suite.png` - 综合相关性分析
- `data_quality_analysis.png` - 数据质量评估
- `advanced_target_analysis.png` - 高级目标分析
- `model_input_analysis.png` - 模型输入分析

#### 5. **`visualization_suite.py`** - 综合可视化套件
**功能**:
- 目标变量配对图
- 特征-目标关系网格
- 分布比较（含KDE）
- 相关性网络可视化
- 数据质量评估图表
- 综合箱线图分析

**输出文件**:
- `target_pairplot.png` - 目标变量成对关系
- `feature_target_relationships.png` - 所有特征-目标散点图
- `distribution_comparison.png` - 分布分析（含KDE）
- `correlation_analysis_suite.png` - 综合相关性分析
- `data_quality_analysis.png` - 数据质量评估
- `comprehensive_boxplots.png` - 详细箱线图比较

#### 6. **`run_all_analyses.py`** - 主控制脚本
**功能**:
- 顺序执行所有分析脚本
- 统一的错误处理
- 进度监控
- 完整性检查

### 自动化脚本

#### **`run_all_analyses.py`** - 一键运行所有分析

```bash
cd statistics
python run_all_analyses.py
```

## 生成的文件结构

### CSV数据文件
- `descriptive_statistics.csv` - 基础描述性统计
- `correlation_matrix.csv` - 完整相关性矩阵
- `pearson_correlation_matrix.csv` - Pearson相关性矩阵
- `spearman_correlation_matrix.csv` - Spearman相关性矩阵
- `feature_target_correlations.csv` - 特征-目标相关性摘要
- `normality_tests.csv` - 正态性检验结果
- `detailed_correlations.csv` - 相关性（含p值）
- `outlier_analysis.csv` - 异常值检测结果
- `pca_analysis.csv` - PCA结果和解释方差
- `pca_loadings.csv` - 主成分特征载荷
- `input_detailed_statistics.csv` - 输入特征详细统计
- `target_detailed_statistics.csv` - 目标变量详细统计
- `feature_importance_scores.csv` - 特征重要性评分

### 可视化文件（PNG）
- `correlation_heatmap.png` - 基础相关性热图
- `correlation_heatmaps.png` - 双重相关性热图
- `target_distributions.png` - 目标变量分布
- `variable_distributions.png` - 所有变量分布
- `target_pairplot.png` - 目标变量配对图
- `input_pairplot.png` - 输入特征配对图
- `feature_target_relationships.png` - 特征-目标关系图
- `distribution_comparison.png` - 分布比较（含KDE）
- `boxplot_comparison.png` - 箱线图比较
- `correlation_analysis_suite.png` - 综合相关性分析
- `data_quality_analysis.png` - 数据质量评估
- `comprehensive_boxplots.png` - 详细箱线图比较
- `pca_analysis_plots.png` - PCA可视化套件
- `advanced_target_analysis.png` - 高级目标分析
- `model_input_analysis.png` - 模型输入分析

### 报告文件（Markdown）
- `ADVANCED_ANALYSIS_SUMMARY.md` - 高级分析摘要
- `VISUALIZATION_SUMMARY.md` - 可视化分析摘要

## 使用说明

### 快速开始
```bash
# 运行所有分析
cd statistics
python run_all_analyses.py
```

### 单独分析
```bash
# 基础分析
python run_analysis.py

# 综合分析
python comprehensive_analysis.py

# 高级分析
python advanced_analysis.py

# 可视化分析
python visualization_analysis.py

# 可视化套件
python visualization_suite.py
```

### 依赖要求

#### Python包
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels plotly
```

#### 必需的包:
- `pandas` - 数据处理和分析
- `numpy` - 数值计算
- `matplotlib` - 基础绘图
- `seaborn` - 统计可视化
- `scipy` - 统计函数
- `scikit-learn` - 机器学习工具
- `statsmodels` - 统计建模
- `plotly` - 交互式可视化

## 数据集信息

分析假设以下数据集结构：

### 输入特征（8个变量）：
- `Temp` - 温度
- `pH` - pH值
- `UVA254` - 254nm处的UV吸光度
- `Cl2` - 氯浓度 (mg/L)
- `NO2-N` - 亚硝酸氮 (mg/L)
- `DOC` - 溶解有机碳 (mg/L)
- `NH4-N` - 铵态氮 (μg/L)
- `Br` - 溴离子浓度 (μg/L)

### 目标变量（5个DBPs）：
- `DCAA (mg/L)` - 二氯乙酸
- `TCAA (ug/L)` - 三氯乙酸
- `BCAA(mg/L)` - 溴氯乙酸
- `HAA5 (ug/L)` - 五种卤乙酸总量
- `HAA9 (ug/L)` - 九种卤乙酸总量

## 关键分析功能

### 1. 描述性统计
- 均值、中位数、众数、标准差
- 偏度和峰度
- 四分位数和范围
- 变异系数
- 缺失值分析

### 2. 相关性分析
- Pearson和Spearman相关性
- 统计显著性检验
- 特征-目标关系强度
- 多重共线性检测

### 3. 分布分析
- 直方图和KDE图
- 正态性检验
- 分布比较
- 异常值识别

### 4. 主成分分析
- 降维分析
- 解释方差分析
- 特征重要性评估
- 成分解释

### 5. 数据质量评估
- 缺失值分析
- 异常值检测（多种方法）
- 数据范围分析
- 变量缩放评估

### 6. 特征重要性
- 互信息评分
- 相关性排名
- 特征选择建议

## 解释指南

### 相关性解释
- |r| > 0.7: 强相关性
- 0.3 < |r| < 0.7: 中等相关性
- |r| < 0.3: 弱相关性
- p < 0.05: 统计显著

### 异常值阈值
- IQR方法: 超出Q1-1.5×IQR或Q3+1.5×IQR的值
- Z-score方法: |z| > 3个标准差

### 正态性评估
- Shapiro-Wilk检验: p > 0.05表示正态分布
- D'Agostino检验: p > 0.05表示正态分布
- 分布图的视觉检查

### PCA指南
- 解释80-95%方差的成分通常足够
- 载荷有助于解释成分含义
- 特征值 > 1传统上被认为是显著的

## 故障排除

### 常见问题
1. **导入错误**: 确保安装了所有必需的包
2. **文件未找到**: 验证data.csv存在于父目录中
3. **内存问题**: 对于大数据集，考虑数据抽样
4. **图表显示问题**: 脚本使用'Agg'后端以兼容服务器

### 性能说明
- 分析运行时间: 典型数据集大小约1-5分钟
- 内存使用: 少于1000个样本的数据集<100MB
- 推荐环境: Python 3.7+

## 分析结果总结

### 当前数据集统计（基于已生成的分析）
- 数据集大小: 66个样本，13个变量
- 输入特征: 8个
- 目标变量: 5个

### 主要发现
- 正态分布变量（Shapiro-Wilk）: 2个
- 正态分布变量（D'Agostino）: 1个
- 每个变量平均异常值（IQR方法）: 1.5个
- 每个变量平均异常值（Z-score方法）: 0.5个
- 80%方差所需成分: 4个
- 95%方差所需成分: 6个
- 显著Pearson相关性（p<0.05）: 12个
- 显著Spearman相关性（p<0.05）: 17个

## 建议和最佳实践

### 数据预处理建议
1. 考虑对非正态变量进行变换
2. 调查两种方法都识别出的异常值
3. 如需要，使用PCA结果进行降维
4. 关注显著相关性进行特征选择

### 模型开发建议
1. 使用特征重要性评分指导特征选择
2. 考虑目标变量间的相关性进行多输出建模
3. 基于分布分析选择合适的模型类型
4. 使用异常值分析结果进行数据清洗

### 进一步分析建议
1. 时间序列分析（如果有时间维度）
2. 非线性关系探索
3. 特征工程基于相关性发现
4. 集成学习方法评估

---

*注意: 所有图表标题和标签均使用英文，符合项目要求。*

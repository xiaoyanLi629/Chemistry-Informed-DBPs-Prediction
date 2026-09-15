"""
综合模型文件 - 整合基础神经网络、经典机器学习模型和KAN模型
用于DBPs（消毒副产品）预测的多种机器学习方法比较
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
import os
import json

# 经典机器学习相关导入
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

# 统计分析
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest, shapiro

# 可视化相关
import seaborn as sns

# KAN模型相关导入（可选，如果不可用则跳过）
try:
    from kan import KAN
    from kan.utils import create_dataset
    from kan.utils import ex_round
    KAN_AVAILABLE = True
except ImportError:
    print("Warning: KAN library not available. KAN model will be skipped.")
    KAN_AVAILABLE = False

# TabNet导入（可选）
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_tabnet not available. TabNet model will be skipped.")
    TABNET_AVAILABLE = False

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BaseNeuralNetwork(torch.nn.Module):
    """基础神经网络模型"""
    
    def __init__(self, n_feature, n_hidden=10, n_output=5):
        super(BaseNeuralNetwork, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden_2 = torch.nn.Linear(n_hidden, 100)
        self.hidden_3 = torch.nn.Linear(100, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x1 = self.hidden_1(x)
        x2 = F.relu(x1)
        x3 = self.hidden_2(x2)
        x4 = F.relu(x3)
        x5 = self.hidden_3(x4)
        x6 = F.relu(x5)
        y = self.predict(x6)
        return y


class TorchDataset(torch.utils.data.Dataset):
    """PyTorch数据集类"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ClassicalMLPredictor:
    """经典机器学习预测器类"""
    
    def __init__(self):
        self.models = {}
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.target_names = None
        self.results = {}
        self.training_history = {}
        
    def initialize_models(self):
        """初始化各种机器学习模型"""
        self.models = {
            'Linear Regression': MultiOutputRegressor(LinearRegression()),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
            'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.1)),
            'Elastic Net': MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5)),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
            'Support Vector Regression': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale')),
            'K-Nearest Neighbors': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
            'Decision Tree': MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, results_dir='results'):
        """训练和评估所有经典ML模型"""
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            try:
                logging.info(f"Training {name}...")
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # 计算指标
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                results[name] = {
                    'Train MSE': train_mse,
                    'Test MSE': test_mse,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Model': model,
                    'Predictions': y_pred_test
                }
                
                # 生成预测结果散点图
                self.plot_prediction_heatmap(y_test, y_pred_test, name, results_dir, timestamp)
                
                logging.info(f"{name} training completed - Test R²: {test_r2:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                
        return results
    
    def plot_prediction_heatmap(self, y_true, y_pred, model_name, results_dir, timestamp):
        """生成预测结果散点图"""
        try:
            target_names = ['DCAA', 'TCAA', 'BCAA', 'HAA5', 'HAA9']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, target in enumerate(target_names):
                # 创建散点图
                axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                # 添加完美预测线 (y=x)
                min_val = min(y_true[:, i].min(), y_pred[:, i].min())
                max_val = max(y_true[:, i].max(), y_pred[:, i].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # 计算R²和RMSE用于显示
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                
                # 添加拟合线
                z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
                p = np.poly1d(z)
                axes[i].plot(y_true[:, i], p(y_true[:, i]), "b--", alpha=0.8, linewidth=1.5, label=f'Fit Line (slope={z[0]:.2f})')
                
                axes[i].set_xlabel('True Values', fontsize=12)
                axes[i].set_ylabel('Predicted Values', fontsize=12)
                axes[i].set_title(f'{target}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}', fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=9)
                
                # 设置坐标轴范围相等
                axes[i].set_xlim(min_val * 0.95, max_val * 1.05)
                axes[i].set_ylim(min_val * 0.95, max_val * 1.05)
                axes[i].set_aspect('equal', adjustable='box')
            
            # 隐藏最后一个子图
            axes[5].set_visible(False)
            
            plt.suptitle(f'{model_name} - Prediction Results (Scatter Plot)', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(results_dir, f'{safe_model_name}_prediction_scatter_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Prediction scatter plot saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error creating prediction scatter plot for {model_name}: {str(e)}")
    
    def save_training_history(self, model_name, history, results_dir, timestamp):
        """保存训练历史"""
        try:
            os.makedirs(results_dir, exist_ok=True)
            history_file = os.path.join(results_dir, f'{model_name}_training_history_{timestamp}.json')
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logging.info(f"Training history saved for {model_name}")
        except Exception as e:
            logging.error(f"Error saving training history for {model_name}: {str(e)}")


class KANPredictor:
    """KAN模型预测器类"""
    
    def __init__(self):
        self.model = None
        self.results = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """训练和评估KAN模型"""
        if not KAN_AVAILABLE:
            logging.warning("KAN library not available, skipping KAN model")
            return {}
            
        try:
            # 转换为PyTorch张量
            x_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            x_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # 创建更简单的KAN模型以避免过拟合
            self.model = KAN(width=[X_train.shape[1], 3, y_train.shape[1]], grid=5, k=3, seed=42)
            
            # 准备数据集
            dataset = {
                'train_input': x_train_tensor,
                'train_label': y_train_tensor,
                'test_input': x_test_tensor,
                'test_label': y_test_tensor
            }
            
            # 训练模型，增加正则化以减少过拟合
            self.model.fit(dataset, opt="LBFGS", steps=50, lamb=0.01)
            
            # 修剪和精细训练，增加正则化
            self.model = self.model.prune()
            self.model.fit(dataset, opt="LBFGS", steps=50, lr=0.001, lamb=0.1)
            
            # 评估
            train_pred = self.model(x_train_tensor).detach().numpy()
            test_pred = self.model(x_test_tensor).detach().numpy()
            
            # 计算指标
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results = {
                'KAN Model': {
                    'Train MSE': train_mse,
                    'Test MSE': test_mse,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Model': self.model,
                    'Predictions': test_pred
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error training KAN model: {str(e)}")
            return {}


class TabNetPredictor:
    """TabNet模型预测器类（Arik & Pfister, 2021）"""

    def __init__(self):
        self.model = None
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, results_dir='results'):
        """训练和评估TabNet模型"""
        if not TABNET_AVAILABLE:
            logging.warning("pytorch_tabnet not available, skipping TabNet model")
            return {}

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.model = TabNetRegressor(
                n_d=16, n_a=16,
                n_steps=5,
                gamma=1.5,
                n_independent=2,
                n_shared=2,
                momentum=0.3,
                mask_type='entmax',
                verbose=0,
                seed=42
            )

            self.model.fit(
                X_train=X_train.astype(np.float32),
                y_train=y_train.astype(np.float32),
                eval_set=[(X_test.astype(np.float32), y_test.astype(np.float32))],
                eval_name=['val'],
                eval_metric=['mse'],
                max_epochs=300,
                patience=50,
                batch_size=16,
                virtual_batch_size=8,
                drop_last=False
            )

            train_pred = self.model.predict(X_train.astype(np.float32))
            test_pred = self.model.predict(X_test.astype(np.float32))

            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            results = {
                'TabNet': {
                    'Train MSE': train_mse,
                    'Test MSE': test_mse,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Model': self.model,
                    'Predictions': test_pred
                }
            }

            logging.info(f"TabNet - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
            return results

        except Exception as e:
            logging.error(f"Error training TabNet model: {str(e)}")
            return {}


# ================================ 可解释模型相关类 ================================

class ChemistryInformedFeatureExtractor(torch.nn.Module):
    """Chemistry-informed feature extractor with chemistry-derived features."""

    # Mapping from feature name to chemistry-based group
    FEATURE_TO_GROUP = {
        'Temp': 'environmental',
        'pH': 'environmental',
        'UVA254': 'organic_matter',
        'DOC': 'organic_matter',
        'Cl2': 'disinfectant',
        'NO2-N': 'nitrogen_species',
        'NH4-N': 'nitrogen_species',
        'Br': 'halides',
    }

    # Canonical group order (used for consistent output ordering)
    GROUP_ORDER = ['environmental', 'organic_matter', 'disinfectant', 'nitrogen_species', 'halides']

    @staticmethod
    def _normalize_feature_name(name):
        """Normalize feature name by stripping units and whitespace."""
        import re
        name = name.strip()
        name = re.sub(r'\s*\(.*\)\s*$', '', name)
        name = name.strip()
        name = name.replace(' -', '-')
        return name

    def __init__(self, input_dim, hidden_dim=32, feature_names=None, x_mean=None, x_std=None,
                 grouping='chemical', rng_seed=None, use_chemistry=True):
        super(ChemistryInformedFeatureExtractor, self).__init__()

        self.hidden_dim = hidden_dim

        # Store scaler params for inverse-transforming to original scale (needed for chemistry features)
        if x_mean is not None and x_std is not None:
            self.register_buffer('x_mean', torch.FloatTensor(x_mean))
            self.register_buffer('x_std', torch.FloatTensor(x_std))

        # Build feature groups from feature_names if provided, else use legacy defaults
        if feature_names is not None:
            self.feature_groups = self._build_feature_groups(feature_names)
        else:
            # Legacy default: original 8-feature dataset (Temp, pH, UVA254, Cl2, NO2-N, DOC, NH4-N, Br)
            self.feature_groups = {
                'environmental': [0, 1],  # Temp, pH
                'organic_matter': [2, 5],  # UVA254, DOC
                'disinfectant': [3],  # Cl2
                'nitrogen_species': [4, 6],  # NO2-N, NH4-N
                'halides': [7]  # Br
            }

        # Ablation variants of the grouping structure
        if grouping == 'none':
            # single undifferentiated group over all input features
            self.feature_groups = {'organic_matter': list(range(input_dim))}
        elif grouping == 'random':
            # same group sizes as the chemical grouping, but features assigned randomly
            sizes = [(g, len(idx)) for g, idx in self.feature_groups.items()]
            perm = list(np.random.default_rng(rng_seed).permutation(input_dim))
            random_groups, pos = {}, 0
            for g, size in sizes:
                random_groups[g] = sorted(int(i) for i in perm[pos:pos + size])
                pos += size
            self.feature_groups = random_groups
        elif grouping != 'chemical':
            raise ValueError(f"Unknown grouping mode: {grouping}")

        # Determine which groups are active (have at least one feature)
        self.active_groups = [g for g in self.GROUP_ORDER if g in self.feature_groups and len(self.feature_groups[g]) > 0]

        # Build per-group extractors dynamically; use hidden_dim for groups with >=2 features, hidden_dim//2 for single-feature groups
        self.group_extractors = torch.nn.ModuleDict()
        self.group_out_dims = {}
        for group in self.active_groups:
            group_size = len(self.feature_groups[group])
            out_dim = hidden_dim if group_size >= 2 else hidden_dim // 2
            self.group_extractors[group] = torch.nn.Sequential(
                torch.nn.Linear(group_size, out_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            self.group_out_dims[group] = out_dim

        # Total combined feature dimension
        self.total_combined_dim = sum(self.group_out_dims[g] for g in self.active_groups)

        # Chemistry-derived features (pH-dependent HOCl speciation, Cl2/DOC ratio)
        self._setup_chemistry_indices(feature_names)
        if not use_chemistry:
            self.ph_idx = self.cl2_idx = self.doc_idx = None
            self.has_hocl = self.has_cl2_doc_ratio = False
        n_chem = self._count_chemistry_features()
        if n_chem > 0:
            self.chemistry_encoder = torch.nn.Sequential(
                torch.nn.Linear(n_chem, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            self.chem_out_dim = hidden_dim // 2
        else:
            self.chemistry_encoder = None
            self.chem_out_dim = 0

        # Interaction extractor uses combined + chemistry features
        interaction_input_dim = self.total_combined_dim + self.chem_out_dim
        self.interaction_extractor = torch.nn.Sequential(
            torch.nn.Linear(interaction_input_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def _setup_chemistry_indices(self, feature_names):
        """Find raw input indices for pH, Cl2, DOC to compute chemistry features."""
        if feature_names is not None:
            norm_to_idx = {self._normalize_feature_name(n): i for i, n in enumerate(feature_names)}
            self.ph_idx = norm_to_idx.get('pH')
            self.cl2_idx = norm_to_idx.get('Cl2')
            self.doc_idx = norm_to_idx.get('DOC')
        else:
            # Legacy defaults for original 8-feature dataset
            self.ph_idx = 1
            self.cl2_idx = 3
            self.doc_idx = 5
        self.has_hocl = self.ph_idx is not None and self.cl2_idx is not None
        self.has_cl2_doc_ratio = self.cl2_idx is not None and self.doc_idx is not None

    def _count_chemistry_features(self):
        n = 0
        if self.has_hocl:
            n += 2  # f_HOCl and effective_Cl2
        if self.has_cl2_doc_ratio:
            n += 1  # Cl2/DOC ratio
        return n

    def _compute_chemistry_features(self, x):
        """Compute physically-motivated chemistry features from raw input."""
        if self._count_chemistry_features() == 0:
            return None

        # Inverse-transform to original scale for meaningful chemistry computations
        if hasattr(self, 'x_mean'):
            x_orig = x * self.x_std + self.x_mean
        else:
            x_orig = x

        features = []
        if self.has_hocl:
            pH = x_orig[:, self.ph_idx]
            # HOCl speciation: f_HOCl = 1/(1 + 10^(pH - pKa)), pKa ≈ 7.5
            # Using sigmoid for numerical stability: sigmoid(-ln10 * (pH - 7.5))
            f_HOCl = torch.sigmoid(-2.303 * (pH - 7.5))
            features.append(f_HOCl.unsqueeze(1))
            # Effective (reactive) chlorine concentration
            Cl2 = x_orig[:, self.cl2_idx]
            effective_Cl2 = Cl2 * f_HOCl
            features.append(effective_Cl2.unsqueeze(1))

        if self.has_cl2_doc_ratio:
            Cl2 = x_orig[:, self.cl2_idx]
            DOC = x_orig[:, self.doc_idx]
            cl2_doc_ratio = Cl2 / (DOC + 1e-8)
            features.append(cl2_doc_ratio.unsqueeze(1))

        chem_raw = torch.cat(features, dim=1)
        return self.chemistry_encoder(chem_raw)

    @classmethod
    def _build_feature_groups(cls, feature_names):
        """Build index-based feature groups from a list of feature names."""
        groups = {}
        for idx, name in enumerate(feature_names):
            normalized = cls._normalize_feature_name(name)
            group = cls.FEATURE_TO_GROUP.get(normalized)
            if group is not None:
                groups.setdefault(group, []).append(idx)
            else:
                logging.warning(f"Feature '{name}' (normalized: '{normalized}') not found in FEATURE_TO_GROUP mapping; skipping.")
        return groups

    def forward(self, x):
        # Extract group features
        group_features = {}
        for group in self.active_groups:
            group_features[group] = self.group_extractors[group](x[:, self.feature_groups[group]])

        # Combined group features (in canonical order)
        combined_features = torch.cat([group_features[g] for g in self.active_groups], dim=1)

        # Chemistry-derived features
        chem_features = self._compute_chemistry_features(x)

        # Interaction extractor uses combined + chemistry
        if chem_features is not None:
            interaction_input = torch.cat([combined_features, chem_features], dim=1)
        else:
            interaction_input = combined_features
        interaction_features = self.interaction_extractor(interaction_input)

        result = {}
        result.update(group_features)
        result['interaction'] = interaction_features
        result['combined'] = combined_features
        if chem_features is not None:
            result['chemistry'] = chem_features
        return result


class AttentionMechanism(torch.nn.Module):
    """Chemistry-aware attention mechanism for group importance weighting."""

    def __init__(self, feature_dim, num_groups=5, active_groups=None, use_attention=True):
        super(AttentionMechanism, self).__init__()
        self.active_groups = active_groups if active_groups is not None else ChemistryInformedFeatureExtractor.GROUP_ORDER
        self.use_attention = use_attention
        self.num_groups = num_groups
        if use_attention:
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, feature_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(feature_dim // 2, num_groups),
                torch.nn.Softmax(dim=1)
            )
        else:
            self.attention = None

    def forward(self, features_dict):
        combined_features = features_dict['combined']

        # Include chemistry features for chemistry-aware attention weight computation
        if 'chemistry' in features_dict:
            attn_input = torch.cat([combined_features, features_dict['chemistry']], dim=1)
        else:
            attn_input = combined_features

        if self.use_attention:
            attention_weights = self.attention(attn_input)
        else:
            # ablation: uniform weights, no learned attention
            attention_weights = torch.full(
                (attn_input.shape[0], self.num_groups), 1.0 / self.num_groups,
                device=attn_input.device)

        # Apply attention weights to group features
        weighted_features = []
        for i, group in enumerate(self.active_groups):
            if group in features_dict:
                weighted = features_dict[group] * attention_weights[:, i:i+1]
                weighted_features.append(weighted)

        return torch.cat(weighted_features, dim=1), attention_weights


class DBPsSpecificPredictor(torch.nn.Module):
    """DBPs特定预测器 - supports variable number of targets"""

    def __init__(self, input_dim, num_targets=5):
        super(DBPsSpecificPredictor, self).__init__()

        # Create one predictor branch per target
        self.predictors = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
            for _ in range(num_targets)
        ])

    def forward(self, x):
        outputs = [predictor(x) for predictor in self.predictors]
        return torch.cat(outputs, dim=1)


class HierarchicalConsistencyLoss(torch.nn.Module):
    """Chemistry-based loss enforcing the HAA aggregate hierarchy.

    HAA5 = MCAA + DCAA + TCAA + MBAA + DBAA and HAA9 = HAA5 + BCAA + BDCAA +
    CDBAA + TBAA. With only DCAA, TCAA (and optionally BCAA) measured, the
    satisfiable constraints are lower bounds: HAA5 >= DCAA + TCAA and
    HAA9 >= HAA5 + BCAA (BCAA is an HAA9 species, not an HAA5 species).
    """

    def __init__(self, target_names, y_mean=None, y_std=None, lambda_h=0.1,
                 use_bcaa=False, nonneg=False):
        super(HierarchicalConsistencyLoss, self).__init__()
        self.lambda_h = lambda_h
        self.use_bcaa = use_bcaa
        self.nonneg = nonneg

        # Find target indices
        self.dcaa_idx = self._find_idx(target_names, 'DCAA')
        self.tcaa_idx = self._find_idx(target_names, 'TCAA')
        self.bcaa_idx = self._find_idx(target_names, 'BCAA')
        self.haa5_idx = self._find_idx(target_names, 'HAA5')
        self.haa9_idx = self._find_idx(target_names, 'HAA9')

        # Store scaler params for original-scale constraint
        if y_mean is not None and y_std is not None:
            self.register_buffer('y_mean', torch.FloatTensor(y_mean))
            self.register_buffer('y_std', torch.FloatTensor(y_std))
            self.has_scaler = True
        else:
            self.has_scaler = False

        self.active = (self.haa5_idx is not None and
                       self.dcaa_idx is not None and
                       self.tcaa_idx is not None)

    @staticmethod
    def _find_idx(target_names, key):
        """Find index by substring match (handles names with units)."""
        for i, name in enumerate(target_names):
            if key in name:
                return i
        return None

    def _species_sum(self, pred):
        """Sum of measured HAA5 species (DCAA + TCAA) that must not exceed HAA5."""
        return pred[:, self.dcaa_idx] + pred[:, self.tcaa_idx]

    def _haa9_lower_bound(self, pred):
        """HAA5 (+ BCAA when measured and use_bcaa) that must not exceed HAA9."""
        total = pred[:, self.haa5_idx]
        if self.use_bcaa and self.bcaa_idx is not None:
            total = total + pred[:, self.bcaa_idx]
        return total

    def forward(self, predictions):
        if not self.active and not self.nonneg:
            return torch.tensor(0.0, device=predictions.device)

        # Convert to original scale for meaningful chemical constraints
        if self.has_scaler:
            pred = predictions * self.y_std + self.y_mean
        else:
            pred = predictions

        loss = torch.tensor(0.0, device=predictions.device)

        if self.active:
            # Constraint 1: HAA5 >= sum of predicted individual species
            violation_haa5 = torch.relu(self._species_sum(pred) - pred[:, self.haa5_idx])
            # Normalize by HAA5's std to keep loss on similar scale as MSE
            if self.has_scaler:
                violation_haa5 = violation_haa5 / (self.y_std[self.haa5_idx] + 1e-8)
            loss = loss + violation_haa5.mean()

            # Constraint 2: HAA9 >= HAA5 + BCAA
            if self.haa9_idx is not None:
                violation_haa9 = torch.relu(
                    self._haa9_lower_bound(pred) - pred[:, self.haa9_idx]
                )
                if self.has_scaler:
                    violation_haa9 = violation_haa9 / (self.y_std[self.haa9_idx] + 1e-8)
                loss = loss + violation_haa9.mean()

        # Constraint 3: all concentrations non-negative
        if self.nonneg:
            neg = torch.relu(-pred)
            if self.has_scaler:
                neg = neg / (self.y_std + 1e-8)
            loss = loss + neg.mean()

        return self.lambda_h * loss

    @torch.no_grad()
    def violation_stats(self, predictions):
        """Fraction of samples violating each active constraint (original scale)."""
        if self.has_scaler:
            pred = predictions * self.y_std + self.y_mean
        else:
            pred = predictions
        stats = {}
        if self.active:
            stats['viol_haa5'] = (self._species_sum(pred) > pred[:, self.haa5_idx] + 1e-9).float().mean().item()
            if self.haa9_idx is not None:
                stats['viol_haa9'] = (self._haa9_lower_bound(pred) > pred[:, self.haa9_idx] + 1e-9).float().mean().item()
        stats['viol_nonneg'] = (pred < -1e-9).any(dim=1).float().mean().item()
        return stats


class ExplainableDBPsModel(torch.nn.Module):
    """Chemistry-informed explainable DBPs prediction model."""

    def __init__(self, input_dim=8, hidden_dim=32, num_targets=5, feature_names=None,
                 x_mean=None, x_std=None, use_attention=True, use_chemistry=True,
                 use_interaction=True, grouping='chemical', rng_seed=None):
        super(ExplainableDBPsModel, self).__init__()

        self.use_interaction = use_interaction
        self.feature_extractor = ChemistryInformedFeatureExtractor(
            input_dim, hidden_dim, feature_names=feature_names,
            x_mean=x_mean, x_std=x_std,
            grouping=grouping, rng_seed=rng_seed, use_chemistry=use_chemistry
        )
        total_features = self.feature_extractor.total_combined_dim
        chem_dim = self.feature_extractor.chem_out_dim
        num_groups = len(self.feature_extractor.active_groups)

        # Attention input includes chemistry features for chemistry-aware weighting
        attention_input_dim = total_features + chem_dim
        self.attention = AttentionMechanism(attention_input_dim, num_groups=num_groups,
                                           active_groups=self.feature_extractor.active_groups,
                                           use_attention=use_attention)

        # Final: attended groups + chemistry + interaction
        final_dim = total_features + chem_dim + (hidden_dim if use_interaction else 0)
        self.predictor = DBPsSpecificPredictor(final_dim, num_targets)

    def forward(self, x, return_attention=False):
        # Feature extraction (groups + chemistry + interaction)
        features = self.feature_extractor(x)

        # Chemistry-aware attention
        attended_features, attention_weights = self.attention(features)

        # Concatenate: attended groups + chemistry + interaction
        parts = [attended_features]
        if 'chemistry' in features:
            parts.append(features['chemistry'])
        if self.use_interaction:
            parts.append(features['interaction'])
        final_features = torch.cat(parts, dim=1)

        # Multi-task prediction
        predictions = self.predictor(final_features)

        if return_attention:
            return predictions, attention_weights, features
        return predictions


class ExplainableModelPredictor:
    """Chemistry-informed explainable model predictor."""

    def __init__(self, feature_names=None, target_names=None):
        self.model = None
        self.results = {}
        self.feature_names = feature_names
        self.target_names = target_names

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, epochs=300, results_dir='results',
                           x_mean=None, x_std=None, y_mean=None, y_std=None):
        """Train and evaluate the chemistry-informed explainable model."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Create model with scaler params for chemistry features
            self.model = ExplainableDBPsModel(
                input_dim=X_train.shape[1],
                num_targets=y_train.shape[1],
                feature_names=self.feature_names,
                x_mean=x_mean, x_std=x_std
            )
            self.model = self.model.to(device)

            # Hierarchical consistency loss (chemistry constraint)
            hierarchy_loss_fn = None
            if self.target_names is not None:
                hierarchy_loss_fn = HierarchicalConsistencyLoss(
                    self.target_names, y_mean=y_mean, y_std=y_std, lambda_h=0.1
                ).to(device)

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)

            # Optimizer and loss
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = torch.nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

            # Training history
            train_losses = []
            test_losses = []
            train_r2_scores = []
            test_r2_scores = []
            learning_rates = []

            # Train
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                predictions = self.model(X_train_tensor)
                mse_loss = criterion(predictions, y_train_tensor)
                # Add chemistry-based hierarchical constraint loss
                if hierarchy_loss_fn is not None:
                    hierarchy_loss = hierarchy_loss_fn(predictions)
                    loss = mse_loss + hierarchy_loss
                else:
                    loss = mse_loss
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                learning_rates.append(optimizer.param_groups[0]['lr'])
                
                # 每10个epoch记录一次
                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        test_pred = self.model(X_test_tensor)
                        test_loss = criterion(test_pred, y_test_tensor)
                        test_losses.append(test_loss.item())
                        
                        # 计算R²分数
                        train_pred_np = self.model(X_train_tensor).cpu().numpy()
                        test_pred_np = test_pred.cpu().numpy()
                        
                        train_r2 = r2_score(y_train, train_pred_np)
                        test_r2 = r2_score(y_test, test_pred_np)
                        
                        train_r2_scores.append(train_r2)
                        test_r2_scores.append(test_r2)
                        
                        logging.info(f"Explainable Model Epoch {epoch}: Train Loss = {loss.item():.6f}, Test Loss = {test_loss.item():.6f}, Test R² = {test_r2:.4f}")
                    
                    scheduler.step(test_loss)
                    self.model.train()
            
            # 评估模型
            self.model.eval()
            with torch.no_grad():
                train_pred = self.model(X_train_tensor).cpu().numpy()
                test_pred = self.model(X_test_tensor).cpu().numpy()
            
            # 计算指标
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # 保存训练历史
            history = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_r2_scores': train_r2_scores,
                'test_r2_scores': test_r2_scores,
                'learning_rates': learning_rates,
                'epochs': epochs,
                'final_metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
            }
            
            self.results['Explainable Model'] = history
            self.save_training_history('Explainable_Model', history, results_dir, timestamp)
            
            # 绘制训练曲线
            self.plot_training_curves(history, 'Explainable Model', results_dir, timestamp)
            
            # 生成预测结果散点图
            self.plot_prediction_heatmap(y_test, test_pred, 'Explainable Model', results_dir, timestamp)
            
            results = {
                'Explainable Model': {
                    'Train MSE': train_mse,
                    'Test MSE': test_mse,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Model': self.model,
                    'Predictions': test_pred,
                    'Training_History': history
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error training Explainable Model: {str(e)}")
            return {}
    
    def plot_training_curves(self, history, model_name, results_dir, timestamp):
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制损失曲线
            epochs_range = range(len(history['train_losses']))
            axes[0, 0].plot(epochs_range, history['train_losses'], 'b-', label='Training Loss', alpha=0.8)
            test_epochs = range(0, len(history['train_losses']), 10)
            axes[0, 0].plot(test_epochs, history['test_losses'], 'r-', label='Test Loss', alpha=0.8)
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 绘制R²分数曲线
            axes[0, 1].plot(test_epochs, history['train_r2_scores'], 'b-', label='Training R²', alpha=0.8)
            axes[0, 1].plot(test_epochs, history['test_r2_scores'], 'r-', label='Test R²', alpha=0.8)
            axes[0, 1].set_title('Model R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 绘制学习率曲线
            axes[1, 0].plot(epochs_range, history['learning_rates'], 'g-', label='Learning Rate', alpha=0.8)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
            
            # 计算并绘制过拟合分析
            if len(history['test_r2_scores']) > 10:
                overfitting_metric = [abs(train_r2 - test_r2) for train_r2, test_r2 in 
                                    zip(history['train_r2_scores'], history['test_r2_scores'])]
                axes[1, 1].plot(test_epochs, overfitting_metric, 'purple', label='|Train R² - Test R²|', alpha=0.8)
                axes[1, 1].set_title('Overfitting Analysis')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('R² Gap')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_name} - Training Curves', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'{model_name.replace(" ", "_")}_training_curve_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Training curve saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error plotting training curves for {model_name}: {str(e)}")
    
    def plot_prediction_heatmap(self, y_true, y_pred, model_name, results_dir, timestamp):
        """生成预测结果散点图"""
        try:
            target_names = ['DCAA', 'TCAA', 'BCAA', 'HAA5', 'HAA9']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, target in enumerate(target_names):
                # 创建散点图
                axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                # 添加完美预测线 (y=x)
                min_val = min(y_true[:, i].min(), y_pred[:, i].min())
                max_val = max(y_true[:, i].max(), y_pred[:, i].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # 计算R²和RMSE用于显示
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                
                # 添加拟合线
                z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
                p = np.poly1d(z)
                axes[i].plot(y_true[:, i], p(y_true[:, i]), "b--", alpha=0.8, linewidth=1.5, label=f'Fit Line (slope={z[0]:.2f})')
                
                axes[i].set_xlabel('True Values', fontsize=12)
                axes[i].set_ylabel('Predicted Values', fontsize=12)
                axes[i].set_title(f'{target}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}', fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=9)
                
                # 设置坐标轴范围相等
                axes[i].set_xlim(min_val * 0.95, max_val * 1.05)
                axes[i].set_ylim(min_val * 0.95, max_val * 1.05)
                axes[i].set_aspect('equal', adjustable='box')
            
            # 隐藏最后一个子图
            axes[5].set_visible(False)
            
            plt.suptitle(f'{model_name} - Prediction Results (Scatter Plot)', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(results_dir, f'{safe_model_name}_prediction_scatter_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Prediction scatter plot saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error creating prediction scatter plot for {model_name}: {str(e)}")
    
    def save_training_history(self, model_name, history, results_dir, timestamp):
        """保存训练历史"""
        try:
            os.makedirs(results_dir, exist_ok=True)
            history_file = os.path.join(results_dir, f'{model_name}_training_history_{timestamp}.json')
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logging.info(f"Training history saved for {model_name}")
        except Exception as e:
            logging.error(f"Error saving training history for {model_name}: {str(e)}")


class AdvancedModelAnalyzer:
    """高级模型分析器，提供深度分析功能"""
    
    def __init__(self):
        self.feature_names = ['Temp', 'pH', 'UVA254', 'Cl2', 'NO2-N', 'DOC', 'NH4-N', 'Br']
        self.target_names = ['DCAA', 'TCAA', 'BCAA', 'HAA5', 'HAA9']
        
    def create_prediction_bins(self, y_true, y_pred, n_bins=5):
        """将连续预测值转换为分类bins用于混淆矩阵分析"""
        bins_true = []
        bins_pred = []
        
        for i in range(y_true.shape[1]):
            # 为每个目标变量创建bins
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            
            true_binned = np.digitize(y_true[:, i], bin_edges) - 1
            pred_binned = np.digitize(y_pred[:, i], bin_edges) - 1
            
            # 确保索引在有效范围内
            true_binned = np.clip(true_binned, 0, n_bins - 1)
            pred_binned = np.clip(pred_binned, 0, n_bins - 1)
            
            bins_true.append(true_binned)
            bins_pred.append(pred_binned)
            
        return bins_true, bins_pred, bin_edges
    
    def plot_confusion_matrices(self, model_results, results_dir, timestamp, y_test=None):
        """为所有模型生成混淆矩阵热图"""
        try:
            n_models = len(model_results)
            n_targets = len(self.target_names)
            
            # 为每个模型创建混淆矩阵
            for model_name, results in model_results.items():
                if 'Predictions' not in results:
                    continue
                    
                y_pred = results['Predictions']
                # 使用真实测试值或模拟数据
                if y_test is not None:
                    y_true = y_test
                else:
                    y_true = y_pred + np.random.normal(0, 0.1, y_pred.shape)  # 模拟真实值
                
                # 创建bins
                bins_true, bins_pred, _ = self.create_prediction_bins(y_true, y_pred, n_bins=4)
                
                fig, axes = plt.subplots(1, n_targets, figsize=(25, 5))  # 增加整体宽度和高度
                if n_targets == 1:
                    axes = [axes]
                
                for i, target_name in enumerate(self.target_names):
                    cm = confusion_matrix(bins_true[i], bins_pred[i], labels=range(4))
                    
                    # 计算百分比
                    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                    
                    # 绘制热图
                    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                               xticklabels=['Low', 'Med-Low', 'Med-High', 'High'],
                               yticklabels=['Low', 'Med-Low', 'Med-High', 'High'],
                               ax=axes[i])
                    axes[i].set_title(f'{target_name}\nConfusion Matrix (%)')
                    axes[i].set_xlabel('Predicted Category')
                    axes[i].set_ylabel('True Category')
                    
                    # 彻底修复Y轴标签位置，避免与刻度重合
                    axes[i].yaxis.set_label_coords(-0.35, 0.5)  # 更大的偏移，将Y轴标签进一步向左移动
                    
                    # 确保Y轴刻度标签正确显示，增加更大的间距
                    axes[i].tick_params(axis='y', labelrotation=0, pad=8)
                    axes[i].tick_params(axis='x', labelrotation=45, pad=5)
                
                plt.suptitle(f'{model_name} - Prediction Accuracy Analysis', fontsize=16, y=1.15)
                plt.tight_layout(pad=15.0)  # 大幅增加子图间距
                plt.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.88, wspace=0.4, hspace=0.3)  # 显式设置子图间距
                plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logging.error(f"Error creating confusion matrices: {str(e)}")
    
    def analyze_residuals(self, model_results, X_test, results_dir, timestamp, y_test=None):
        """残差分析"""
        try:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            axes = axes.flatten()
            
            model_idx = 0
            for model_name, results in list(model_results.items())[:9]:  # 只分析前9个模型
                if 'Predictions' not in results:
                    continue
                
                y_pred = results['Predictions']
                # 使用真实测试值或模拟数据
                if y_test is not None:
                    y_true = y_test
                else:
                    y_true = y_pred + np.random.normal(0, 0.1, y_pred.shape)
                
                # 计算残差
                residuals = y_true - y_pred
                mean_residuals = np.mean(residuals, axis=1)
                
                # Q-Q图检验正态性
                stats.probplot(mean_residuals, dist="norm", plot=axes[model_idx])
                axes[model_idx].set_title(f'{model_name}\nResidual Q-Q Plot')
                axes[model_idx].grid(True, alpha=0.3)
                
                model_idx += 1
            
            # 隐藏未使用的子图
            for i in range(model_idx, 9):
                axes[i].set_visible(False)
            
            plt.suptitle('Residual Analysis - Normality Tests', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'residual_analysis_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error in residual analysis: {str(e)}")
    
    def feature_importance_analysis(self, model_results, X_train, results_dir, timestamp):
        """特征重要性分析"""
        try:
            importance_data = {}
            
            for model_name, results in model_results.items():
                if 'Model' not in results:
                    continue
                    
                model = results['Model']
                
                # 尝试提取特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance_data[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # 对于线性模型，使用系数的绝对值
                    if hasattr(model, 'estimators_'):  # MultiOutputRegressor
                        coefs = np.mean([est.coef_ for est in model.estimators_], axis=0)
                    else:
                        coefs = np.mean(np.abs(model.coef_), axis=0)
                    importance_data[model_name] = np.abs(coefs)
            
            if importance_data:
                # 创建特征重要性对比图
                df_importance = pd.DataFrame(importance_data, index=self.feature_names)
                
                plt.figure(figsize=(14, 8))
                sns.heatmap(df_importance.T, annot=True, fmt='.3f', cmap='RdYlBu_r')
                plt.title('Feature Importance Comparison Across Models')
                plt.xlabel('Features')
                plt.ylabel('Models')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'feature_importance_{timestamp}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 保存数据
                df_importance.to_csv(os.path.join(results_dir, f'feature_importance_{timestamp}.csv'))
                
        except Exception as e:
            logging.error(f"Error in feature importance analysis: {str(e)}")
    
    def prediction_distribution_analysis(self, model_results, results_dir, timestamp):
        """预测分布分析"""
        try:
            n_models = len(model_results)
            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            model_idx = 0
            for model_name, results in model_results.items():
                if 'Predictions' not in results:
                    continue
                
                row = model_idx // n_cols
                col = model_idx % n_cols
                
                y_pred = results['Predictions']
                
                # 为每个目标变量绘制分布
                for i, target_name in enumerate(self.target_names):
                    axes[row, col].hist(y_pred[:, i], bins=15, alpha=0.7, 
                                       label=target_name, density=True)
                
                axes[row, col].set_title(f'{model_name}\nPrediction Distributions')
                axes[row, col].set_xlabel('Predicted Values')
                axes[row, col].set_ylabel('Density')
                axes[row, col].legend(fontsize=8)
                axes[row, col].grid(True, alpha=0.3)
                
                model_idx += 1
            
            # 隐藏未使用的子图
            for i in range(model_idx, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'prediction_distributions_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error in prediction distribution analysis: {str(e)}")
    
    def model_correlation_analysis(self, model_results, results_dir, timestamp):
        """模型预测相关性分析"""
        try:
            # 收集所有模型的预测结果
            predictions_dict = {}
            for model_name, results in model_results.items():
                if 'Predictions' in results:
                    # 计算每个样本的平均预测值
                    mean_pred = np.mean(results['Predictions'], axis=1)
                    predictions_dict[model_name] = mean_pred
            
            if len(predictions_dict) < 2:
                return
            
            # 创建相关性矩阵
            df_predictions = pd.DataFrame(predictions_dict)
            correlation_matrix = df_predictions.corr()
            
            # 绘制相关性热图
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                       cmap='coolwarm', center=0, square=True, linewidths=0.5)
            plt.title('Model Prediction Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'model_correlation_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存相关性数据
            correlation_matrix.to_csv(os.path.join(results_dir, f'model_correlation_{timestamp}.csv'))
            
        except Exception as e:
            logging.error(f"Error in model correlation analysis: {str(e)}")
    
    def comprehensive_model_analysis(self, model_results, X_train, X_test, results_dir, timestamp, y_test=None):
        """运行所有高级分析"""
        logging.info("Running comprehensive model analysis...")
        
        # 创建高级分析目录
        advanced_dir = os.path.join(results_dir, 'advanced_analysis')
        os.makedirs(advanced_dir, exist_ok=True)
        
        # 运行各种分析
        self.plot_confusion_matrices(model_results, advanced_dir, timestamp, y_test)
        self.analyze_residuals(model_results, X_test, advanced_dir, timestamp, y_test)
        self.feature_importance_analysis(model_results, X_train, advanced_dir, timestamp)
        self.prediction_distribution_analysis(model_results, advanced_dir, timestamp)
        self.model_correlation_analysis(model_results, advanced_dir, timestamp)
        
        logging.info("Advanced analysis completed")


class AblationExperiment:
    """消融实验类 - 特征消融研究"""

    # Legacy default feature names for the original dataset
    DEFAULT_FEATURE_NAMES = ['Temp', 'pH ', 'UVA254(cm)', 'Cl2 (mg/L)',
                             'NO2 -N (mg/L)', 'DOC (mg/L)', 'NH4-N (ug/L)', 'Br (ug/L)']

    def __init__(self, feature_names=None):
        self.models = {}
        self.results = {}
        self.feature_names = feature_names if feature_names is not None else self.DEFAULT_FEATURE_NAMES
        
    def create_ablation_models(self):
        """创建消融模型 - 基于可解释模型，每次移除一个特征"""
        
        # 基础模型（所有特征）- Explainable Model
        self.models['Baseline_All_Features_Explainable'] = {
            'model_type': 'explainable',
            'removed_feature_idx': None,
            'removed_feature_name': None
        }
        
        # 移除单个特征的模型 - Explainable Model
        for i, feature_name in enumerate(self.feature_names):
            model_name = f'Remove_{feature_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}_Explainable'
            self.models[model_name] = {
                'model_type': 'explainable',
                'removed_feature_idx': i,
                'removed_feature_name': feature_name
            }
        
        logging.info(f"Created {len(self.models)} explainable ablation models")
        
    def run_ablation_study(self, X_train, X_test, y_train, y_test, 
                          results_dir='ablation_results', epochs=300):
        """运行消融实验"""
        os.makedirs(results_dir, exist_ok=True)
        logging.info("Starting ablation study...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ablation_results = {}
        
        # 训练基础模型（所有特征）- Explainable Model
        logging.info("Training baseline Explainable model with all features...")
        baseline_explainable_metrics = self.train_explainable_model_ablation(
            X_train, X_test, y_train, y_test, 
            'Baseline_All_Features_Explainable', None, results_dir, timestamp, epochs=300
        )
        baseline_explainable_metrics['Feature_Count'] = X_train.shape[1]
        baseline_explainable_metrics['Model_Type'] = 'Explainable Model'
        
        ablation_results['Baseline_All_Features_Explainable'] = baseline_explainable_metrics
        logging.info(f"Baseline Explainable model - Test R²: {baseline_explainable_metrics['Test R²']:.4f}")
        
        # 训练特征移除模型
        for model_name, model_info in self.models.items():
            if model_name == 'Baseline_All_Features_Explainable':
                continue
                
            logging.info(f"Training {model_name}...")
            
            # 训练可解释模型（移除特定特征）
            metrics = self.train_explainable_model_ablation(
                X_train, X_test, y_train, y_test, 
                model_name, model_info['removed_feature_idx'], results_dir, timestamp, epochs=300
            )
            
            metrics['Removed_Feature'] = model_info['removed_feature_name']
            metrics['Feature_Count'] = X_train.shape[1] - 1 if model_info['removed_feature_idx'] is not None else X_train.shape[1]
            metrics['Model_Type'] = 'Explainable Model'
            metrics['Performance_Drop_R2'] = baseline_explainable_metrics['Test R²'] - metrics['Test R²']
            metrics['Performance_Drop_MSE'] = metrics['Test MSE'] - baseline_explainable_metrics['Test MSE']
            
            ablation_results[model_name] = metrics
            logging.info(f"{model_name} - Test R²: {metrics['Test R²']:.4f} "
                        f"(Drop: {metrics['Performance_Drop_R2']:.4f})")
        
        # 保存消融研究结果
        self.save_ablation_results(ablation_results, results_dir, timestamp)
        
        # 生成消融研究可视化
        self.plot_ablation_analysis(ablation_results, results_dir, timestamp)
        
        # 生成消融研究报告
        self.generate_ablation_report(ablation_results, results_dir, timestamp)
        
        logging.info("Ablation study completed")
        return ablation_results
    
    def save_ablation_results(self, results, results_dir, timestamp):
        """保存消融研究结果"""
        try:
            # 保存为CSV
            results_data = []
            for model_name, metrics in results.items():
                row = {
                    'Model': model_name,
                    'Removed_Feature': metrics.get('Removed_Feature', 'None'),
                    'Feature_Count': metrics['Feature_Count'],
                    'Train_R2': metrics['Train R²'],
                    'Test_R2': metrics['Test R²'],
                    'Train_MSE': metrics['Train MSE'],
                    'Test_MSE': metrics['Test MSE'],
                    'Train_MAE': metrics['Train MAE'],
                    'Test_MAE': metrics['Test MAE'],
                    'Performance_Drop_R2': metrics.get('Performance_Drop_R2', 0),
                    'Performance_Drop_MSE': metrics.get('Performance_Drop_MSE', 0)
                }
                results_data.append(row)
            
            df_results = pd.DataFrame(results_data)
            csv_file = os.path.join(results_dir, f'ablation_results_{timestamp}.csv')
            df_results.to_csv(csv_file, index=False)
            logging.info(f"Ablation results saved to {csv_file}")
            
            # 保存为JSON（包含预测结果）
            json_results = {}
            for model_name, metrics in results.items():
                json_results[model_name] = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in metrics.items()
                }
            
            json_file = os.path.join(results_dir, f'ablation_detailed_{timestamp}.json')
            with open(json_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            logging.info(f"Detailed ablation results saved to {json_file}")
            
        except Exception as e:
            logging.error(f"Error saving ablation results: {str(e)}")
    
    def plot_ablation_analysis(self, results, results_dir, timestamp):
        """生成消融研究可视化"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Ablation Study Analysis', fontsize=16, fontweight='bold')
            
            # 准备数据
            model_names = []
            removed_features = []
            test_r2_scores = []
            performance_drops = []
            
            baseline_r2 = results['Baseline_All_Features_Explainable']['Test R²']
            
            for model_name, metrics in results.items():
                if model_name == 'Baseline_All_Features_Explainable':
                    continue
                model_names.append(model_name.replace('Remove_', '').replace('_', ' '))
                removed_features.append(metrics['Removed_Feature'])
                test_r2_scores.append(metrics['Test R²'])
                performance_drops.append(metrics['Performance_Drop_R2'])
            
            # 1. 特征重要性（基于性能下降）
            sorted_indices = np.argsort(performance_drops)[::-1]  # 降序排列
            sorted_features = [removed_features[i] for i in sorted_indices]
            sorted_drops = [performance_drops[i] for i in sorted_indices]
            
            bars1 = axes[0, 0].barh(sorted_features, sorted_drops)
            axes[0, 0].set_title('Feature Importance (Performance Drop when Removed)')
            axes[0, 0].set_xlabel('R² Score Drop')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 为最重要的特征着色
            if len(bars1) > 0:
                bars1[0].set_color('red')
            if len(bars1) > 1:
                bars1[1].set_color('orange')
            
            # 2. 移除特征后的R²分数
            axes[0, 1].barh(removed_features, test_r2_scores)
            axes[0, 1].axvline(x=baseline_r2, color='red', linestyle='--', 
                              label=f'Baseline (All Features): {baseline_r2:.4f}')
            axes[0, 1].set_title('Test R² Score after Feature Removal')
            axes[0, 1].set_xlabel('R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 性能下降分布
            axes[1, 0].hist(performance_drops, bins=max(len(performance_drops)//2, 1), alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Distribution of Performance Drops')
            axes[1, 0].set_xlabel('R² Score Drop')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 特征重要性排名
            importance_ranking = list(zip(sorted_features, sorted_drops))
            y_pos = np.arange(len(importance_ranking))
            
            axes[1, 1].barh(y_pos, [drop for _, drop in importance_ranking])
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels([feat for feat, _ in importance_ranking])
            axes[1, 1].set_title('Feature Importance Ranking')
            axes[1, 1].set_xlabel('Performance Drop (R² Score)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = os.path.join(results_dir, f'ablation_analysis_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logging.info(f"Ablation analysis plot saved to {plot_file}")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating ablation analysis plot: {str(e)}")
    
    def generate_ablation_report(self, results, results_dir, timestamp):
        """生成消融研究报告"""
        try:
            baseline_r2 = results['Baseline_All_Features_Explainable']['Test R²']
            
            # 找到最重要的特征（移除时性能下降最大）
            max_drop = 0
            most_important_feature = ""
            least_important_feature = ""
            min_drop = float('inf')
            
            for model_name, metrics in results.items():
                if model_name == 'Baseline_All_Features_Explainable':
                    continue
                drop = metrics['Performance_Drop_R2']
                if drop > max_drop:
                    max_drop = drop
                    most_important_feature = metrics['Removed_Feature']
                if drop < min_drop:
                    min_drop = drop
                    least_important_feature = metrics['Removed_Feature']
            
            report = f"""
# 消融研究报告 (Ablation Study Report)

## 实验设置
- 基准模型: Explainable Model (300 epochs)
- 消融方法: 单特征移除法 (Leave-One-Out)
- 评估指标: R² Score, MSE, MAE
- 实验时间: {timestamp}

## 基准模型性能
- 使用所有特征的基准模型性能:
  - Test R²: {baseline_r2:.4f}
  - Test MSE: {results['Baseline_All_Features_Explainable']['Test MSE']:.4f}
  - Test MAE: {results['Baseline_All_Features_Explainable']['Test MAE']:.4f}

## 特征重要性分析

### 最重要特征
- **{most_important_feature}**: 移除时R²下降 {max_drop:.4f}
- 这是对模型性能影响最大的特征

### 最不重要特征  
- **{least_important_feature}**: 移除时R²下降 {min_drop:.4f}
- 这是对模型性能影响最小的特征

## 详细结果

| 移除的特征 | Test R² | R²下降 | Test MSE | MSE变化 |
|------------|---------|--------|----------|---------|
"""
            
            # 按性能下降排序
            sorted_results = sorted(
                [(name, metrics) for name, metrics in results.items() if name != 'Baseline_All_Features_Explainable'],
                key=lambda x: x[1]['Performance_Drop_R2'],
                reverse=True
            )
            
            for model_name, metrics in sorted_results:
                report += f"| {metrics['Removed_Feature']} | {metrics['Test R²']:.4f} | {metrics['Performance_Drop_R2']:.4f} | {metrics['Test MSE']:.4f} | {metrics['Performance_Drop_MSE']:.4f} |\n"
            
            report += f"""

## 总结
1. 共测试了 {len(self.feature_names)} 个特征的重要性
2. 性能下降范围: {min_drop:.4f} - {max_drop:.4f} (R² Score)
3. 平均性能下降: {np.mean([metrics['Performance_Drop_R2'] for name, metrics in results.items() if name != 'Baseline_All_Features_Explainable']):.4f}

## 建议
- 考虑移除影响最小的特征: {least_important_feature}
- 重点关注最重要的特征: {most_important_feature}
- 可以考虑特征选择来简化模型
"""
            
            # 保存报告
            report_file = os.path.join(results_dir, f'ablation_report_{timestamp}.md')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logging.info(f"Ablation report saved to {report_file}")
            
        except Exception as e:
            logging.error(f"Error generating ablation report: {str(e)}")
    
    def train_explainable_model_ablation(self, X_train, X_test, y_train, y_test, 
                                       model_name, removed_feature_idx, results_dir, timestamp, epochs=300):
        """为消融研究训练可解释模型并保存训练过程"""
        try:
            # 如果需要移除特征，创建新的数据集
            if removed_feature_idx is not None:
                logging.info(f"Removing feature at index {removed_feature_idx} from {X_train.shape[1]} features")
                X_train_model = np.delete(X_train, removed_feature_idx, axis=1)
                X_test_model = np.delete(X_test, removed_feature_idx, axis=1)
                input_dim = X_train_model.shape[1]
                logging.info(f"After removal: X_train shape {X_train_model.shape}, input_dim {input_dim}")
            else:
                X_train_model = X_train.copy()
                X_test_model = X_test.copy()
                input_dim = X_train.shape[1]
                logging.info(f"Using all features: X_train shape {X_train_model.shape}, input_dim {input_dim}")
            
            # 转换为PyTorch张量
            x_train_tensor = torch.FloatTensor(X_train_model)
            y_train_tensor = torch.FloatTensor(y_train)
            x_test_tensor = torch.FloatTensor(X_test_model)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # 创建数据集和数据加载器
            train_dataset = TorchDataset(x_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
            
            # 创建简化的可解释模型（用于消融研究）
            model = SimpleExplainableModel(input_dim=input_dim, hidden_dim=64, num_targets=y_train.shape[1])
            
            # 定义优化器和损失函数
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            loss_func = torch.nn.MSELoss()
            
            # 训练历史记录
            train_losses = []
            test_losses = []
            train_r2_scores = []
            test_r2_scores = []
            learning_rates = []
            
            # 训练模型
            model.train()
            for epoch in range(epochs):
                epoch_train_loss = 0
                batch_count = 0
                
                for batch_x, batch_y in train_loader:
                    prediction = model(batch_x)
                    loss = loss_func(prediction, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    batch_count += 1
                
                # 计算平均训练损失
                avg_train_loss = epoch_train_loss / batch_count
                train_losses.append(avg_train_loss)
                
                # 每10个epoch评估一次
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        train_pred = model(x_train_tensor).numpy()
                        test_pred = model(x_test_tensor).numpy()
                        
                        test_loss = loss_func(model(x_test_tensor), y_test_tensor).item()
                        test_losses.append(test_loss)
                        
                        train_r2 = r2_score(y_train, train_pred)
                        test_r2 = r2_score(y_test, test_pred)
                        
                        train_r2_scores.append(train_r2)
                        test_r2_scores.append(test_r2)
                        learning_rates.append(optimizer.param_groups[0]['lr'])
                    
                    logging.info(f"Ablation {model_name} Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_loss:.6f}, Test R² = {test_r2:.4f}")
                    
                    # 学习率调度
                    scheduler.step(test_loss)
                    model.train()
            
            # 最终评估
            model.eval()
            with torch.no_grad():
                train_pred = model(x_train_tensor).numpy()
                test_pred = model(x_test_tensor).numpy()
            
            # 计算指标
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # 保存训练历史
            history = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_r2_scores': train_r2_scores,
                'test_r2_scores': test_r2_scores,
                'learning_rates': learning_rates,
                'epochs': epochs,
                'final_metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
            }
            
            # 保存训练历史到消融研究目录
            self.save_ablation_training_history(model_name, history, results_dir, timestamp)
            
            # 绘制训练曲线
            self.plot_ablation_training_curves(history, model_name, results_dir, timestamp)
            
            # 生成预测结果散点图
            self.plot_ablation_prediction_scatter(y_test, test_pred, model_name, results_dir, timestamp)
            
            return {
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Predictions': test_pred,
                'Training_History': history
            }
            
        except Exception as e:
            logging.error(f"Error training explainable model for ablation {model_name}: {str(e)}")
            return {
                'Train MSE': float('inf'),
                'Test MSE': float('inf'),
                'Train MAE': float('inf'),
                'Test MAE': float('inf'),
                'Train R²': 0.0,
                'Test R²': 0.0,
                'Predictions': np.zeros((y_test.shape[0], y_test.shape[1])),
                'Training_History': {}
            }
    
    def save_ablation_training_history(self, model_name, history, results_dir, timestamp):
        """保存消融研究的训练历史"""
        try:
            os.makedirs(results_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            history_file = os.path.join(results_dir, f'{safe_model_name}_training_history_{timestamp}.json')
            
            # 转换numpy数组为列表以便JSON序列化
            json_history = {}
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    json_history[key] = value.tolist()
                elif isinstance(value, dict):
                    json_history[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
                else:
                    json_history[key] = value
            
            with open(history_file, 'w') as f:
                json.dump(json_history, f, indent=2)
            
            logging.info(f"Ablation training history saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error saving ablation training history for {model_name}: {str(e)}")
    
    def plot_ablation_training_curves(self, history, model_name, results_dir, timestamp):
        """绘制消融研究的训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制损失曲线
            epochs_range = range(len(history['train_losses']))
            test_epochs = range(0, len(history['train_losses']), 10)
            
            axes[0, 0].plot(epochs_range, history['train_losses'], 'b-', label='Training Loss', alpha=0.8)
            axes[0, 0].plot(test_epochs, history['test_losses'], 'r-', label='Test Loss', alpha=0.8)
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 绘制R²分数曲线
            axes[0, 1].plot(test_epochs, history['train_r2_scores'], 'b-', label='Training R²', alpha=0.8)
            axes[0, 1].plot(test_epochs, history['test_r2_scores'], 'r-', label='Test R²', alpha=0.8)
            axes[0, 1].set_title('Model R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 绘制对数尺度损失
            axes[1, 0].semilogy(epochs_range, history['train_losses'], 'b-', label='Training Loss', alpha=0.8)
            axes[1, 0].semilogy(test_epochs, history['test_losses'], 'r-', label='Test Loss', alpha=0.8)
            axes[1, 0].set_title('Model Loss (Log Scale)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (log scale)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 绘制学习率变化
            axes[1, 1].plot(test_epochs, history['learning_rates'], 'g-', label='Learning Rate', alpha=0.8)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Ablation Study Training Curves - {model_name}', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(results_dir, f'{safe_model_name}_training_curve_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Ablation training curve saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error creating ablation training curve for {model_name}: {str(e)}")
    
    def plot_ablation_prediction_scatter(self, y_true, y_pred, model_name, results_dir, timestamp):
        """绘制消融研究的预测结果散点图"""
        try:
            target_names = ['DCAA', 'TCAA', 'BCAA', 'HAA5', 'HAA9']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, target in enumerate(target_names):
                # 创建散点图
                axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                # 添加完美预测线 (y=x)
                min_val = min(y_true[:, i].min(), y_pred[:, i].min())
                max_val = max(y_true[:, i].max(), y_pred[:, i].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # 计算R²和RMSE用于显示
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                
                # 添加拟合线
                z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
                p = np.poly1d(z)
                axes[i].plot(y_true[:, i], p(y_true[:, i]), "b--", alpha=0.8, linewidth=1.5, label=f'Fit Line (slope={z[0]:.2f})')
                
                axes[i].set_xlabel('True Values', fontsize=12)
                axes[i].set_ylabel('Predicted Values', fontsize=12)
                axes[i].set_title(f'{target}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}', fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=9)
                
                # 设置坐标轴范围相等
                axes[i].set_xlim(min_val * 0.95, max_val * 1.05)
                axes[i].set_ylim(min_val * 0.95, max_val * 1.05)
                axes[i].set_aspect('equal', adjustable='box')
            
            # 隐藏最后一个子图
            axes[5].set_visible(False)
            
            plt.suptitle(f'Ablation Study Prediction Results - {model_name}', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(results_dir, f'{safe_model_name}_prediction_scatter_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Ablation prediction scatter plot saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error creating ablation prediction scatter plot for {model_name}: {str(e)}")


class NeuralNetworkPredictor:
    """神经网络预测器类"""
    
    def __init__(self):
        self.model = None
        self.results = {}
        self.training_history = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, epochs=300, results_dir='results'):
        """训练和评估神经网络模型"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 转换为PyTorch张量
            x_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            x_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # 创建数据集和数据加载器
            train_dataset = TorchDataset(x_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
            
            # 创建模型
            self.model = BaseNeuralNetwork(X_train.shape[1], n_hidden=10, n_output=y_train.shape[1])
            
            # 定义优化器和损失函数
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            loss_func = torch.nn.MSELoss()
            
            # 训练历史记录
            train_losses = []
            test_losses = []
            train_r2_scores = []
            test_r2_scores = []
            
            # 训练模型
            self.model.train()
            for epoch in range(epochs):
                epoch_train_loss = 0
                batch_count = 0
                
                for batch_x, batch_y in train_loader:
                    prediction = self.model(batch_x)
                    loss = loss_func(prediction, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    batch_count += 1
                
                # 计算平均训练损失
                avg_train_loss = epoch_train_loss / batch_count
                train_losses.append(avg_train_loss)
                
                # 每10个epoch评估一次
                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        train_pred = self.model(x_train_tensor).numpy()
                        test_pred = self.model(x_test_tensor).numpy()
                        
                        test_loss = loss_func(self.model(x_test_tensor), y_test_tensor).item()
                        test_losses.append(test_loss)
                        
                        train_r2 = r2_score(y_train, train_pred)
                        test_r2 = r2_score(y_test, test_pred)
                        
                        train_r2_scores.append(train_r2)
                        test_r2_scores.append(test_r2)
                    
                    logging.info(f"Neural Network Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_loss:.6f}, Test R² = {test_r2:.4f}")
                    self.model.train()
            
            # 最终评估
            self.model.eval()
            with torch.no_grad():
                train_pred = self.model(x_train_tensor).numpy()
                test_pred = self.model(x_test_tensor).numpy()
            
            # 计算指标
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # 保存训练历史
            history = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_r2_scores': train_r2_scores,
                'test_r2_scores': test_r2_scores,
                'epochs': epochs,
                'final_metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
            }
            
            self.training_history['Neural Network'] = history
            self.save_training_history('Neural_Network', history, results_dir, timestamp)
            
            # 绘制训练曲线
            self.plot_training_curves(history, 'Neural Network', results_dir, timestamp)
            
            # 生成预测结果散点图
            self.plot_prediction_heatmap(y_test, test_pred, 'Neural Network', results_dir, timestamp)
            
            results = {
                'Neural Network': {
                    'Train MSE': train_mse,
                    'Test MSE': test_mse,
                    'Train MAE': train_mae,
                    'Test MAE': test_mae,
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Model': self.model,
                    'Predictions': test_pred,
                    'Training_History': history
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error training Neural Network: {str(e)}")
            return {}
    
    def plot_training_curves(self, history, model_name, results_dir, timestamp):
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制损失曲线
            epochs_range = range(len(history['train_losses']))
            axes[0, 0].plot(epochs_range, history['train_losses'], 'b-', label='Training Loss', alpha=0.8)
            test_epochs = range(0, len(history['train_losses']), 10)
            axes[0, 0].plot(test_epochs, history['test_losses'], 'r-', label='Test Loss', alpha=0.8)
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 绘制R²分数曲线
            axes[0, 1].plot(test_epochs, history['train_r2_scores'], 'b-', label='Training R²', alpha=0.8)
            axes[0, 1].plot(test_epochs, history['test_r2_scores'], 'r-', label='Test R²', alpha=0.8)
            axes[0, 1].set_title('Model R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 绘制对数尺度损失
            axes[1, 0].semilogy(epochs_range, history['train_losses'], 'b-', label='Training Loss', alpha=0.8)
            axes[1, 0].semilogy(test_epochs, history['test_losses'], 'r-', label='Test Loss', alpha=0.8)
            axes[1, 0].set_title('Model Loss (Log Scale)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (log scale)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 计算并绘制过拟合分析
            if len(history['test_r2_scores']) > 10:
                overfitting_metric = [abs(train_r2 - test_r2) for train_r2, test_r2 in 
                                    zip(history['train_r2_scores'], history['test_r2_scores'])]
                axes[1, 1].plot(test_epochs, overfitting_metric, 'g-', label='|Train R² - Test R²|', alpha=0.8)
                axes[1, 1].set_title('Overfitting Analysis')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('R² Gap')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_name} - Training Curves', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'{model_name.replace(" ", "_")}_training_curve_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Training curve saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error plotting training curves for {model_name}: {str(e)}")
    
    def plot_prediction_heatmap(self, y_true, y_pred, model_name, results_dir, timestamp):
        """生成预测结果散点图"""
        try:
            target_names = ['DCAA', 'TCAA', 'BCAA', 'HAA5', 'HAA9']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, target in enumerate(target_names):
                # 创建散点图
                axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                
                # 添加完美预测线 (y=x)
                min_val = min(y_true[:, i].min(), y_pred[:, i].min())
                max_val = max(y_true[:, i].max(), y_pred[:, i].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # 计算R²和RMSE用于显示
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                
                # 添加拟合线
                z = np.polyfit(y_true[:, i], y_pred[:, i], 1)
                p = np.poly1d(z)
                axes[i].plot(y_true[:, i], p(y_true[:, i]), "b--", alpha=0.8, linewidth=1.5, label=f'Fit Line (slope={z[0]:.2f})')
                
                axes[i].set_xlabel('True Values', fontsize=12)
                axes[i].set_ylabel('Predicted Values', fontsize=12)
                axes[i].set_title(f'{target}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}', fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(fontsize=9)
                
                # 设置坐标轴范围相等
                axes[i].set_xlim(min_val * 0.95, max_val * 1.05)
                axes[i].set_ylim(min_val * 0.95, max_val * 1.05)
                axes[i].set_aspect('equal', adjustable='box')
            
            # 隐藏最后一个子图
            axes[5].set_visible(False)
            
            plt.suptitle(f'{model_name} - Prediction Results (Scatter Plot)', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(results_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(results_dir, f'{safe_model_name}_prediction_scatter_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Prediction scatter plot saved for {model_name}")
            
        except Exception as e:
            logging.error(f"Error creating prediction scatter plot for {model_name}: {str(e)}")
    
    def save_training_history(self, model_name, history, results_dir, timestamp):
        """保存训练历史"""
        try:
            os.makedirs(results_dir, exist_ok=True)
            history_file = os.path.join(results_dir, f'{model_name}_training_history_{timestamp}.json')
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logging.info(f"Training history saved for {model_name}")
        except Exception as e:
            logging.error(f"Error saving training history for {model_name}: {str(e)}")


class ComprehensiveModelEvaluator:
    """综合模型评估器"""

    def __init__(self, data_path='data/data.csv', results_dir='results',
                 target_names=None, feature_names=None):
        self.data_path = data_path
        self.results_dir = results_dir
        self.all_results = {}
        self.feature_names = feature_names  # Will be auto-detected if None
        self.target_names = target_names    # Will be auto-detected if None
        # Track whether feature_names were explicitly provided (for chemistry-based grouping)
        self._explicit_feature_names = feature_names is not None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        log_file = os.path.join(self.results_dir, f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        logging.info("Loading and preprocessing data...")

        # 读取数据
        df = pd.read_csv(self.data_path)
        if 'Sample' in df.columns:
            df = df.drop('Sample', axis=1)

        # 分离特征和目标变量
        if self.target_names is not None and self.feature_names is not None:
            # Use explicitly provided column names
            y = df[self.target_names].values
            X = df[self.feature_names].values
        else:
            # Legacy auto-detection: first 5 cols = targets, rest = features
            self.target_names = df.columns[:5].tolist()
            self.feature_names = df.columns[5:].tolist()
            X = df.iloc[:, 5:].values
            y = df.iloc[:, :5].values
        
        logging.info(f"Features: {self.feature_names}")
        logging.info(f"Targets: {self.target_names}")
        logging.info(f"Data shape - X: {X.shape}, y: {y.shape}")
        
        # 数据标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.3, random_state=42
        )
        
        logging.info(f"Training set: {X_train.shape[0]} samples")
        logging.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test
        
    def run_all_models(self, include_explainable=True, run_ablation=False):
        """运行所有模型"""
        logging.info("Starting comprehensive model evaluation...")
        
        # 加载数据
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # 1. 经典机器学习模型
        logging.info("Training Classical ML models...")
        classical_predictor = ClassicalMLPredictor()
        classical_predictor.initialize_models()
        classical_results = classical_predictor.train_and_evaluate(X_train, X_test, y_train, y_test, self.results_dir)
        self.all_results.update(classical_results)
        
        # 2. 神经网络模型 (300 epochs)
        logging.info("Training Neural Network model...")
        nn_predictor = NeuralNetworkPredictor()
        nn_results = nn_predictor.train_and_evaluate(X_train, X_test, y_train, y_test, epochs=300, results_dir=self.results_dir)
        self.all_results.update(nn_results)
        
        # 3. 可解释模型 (300 epochs)
        if include_explainable:
            logging.info("Training Explainable Model...")
            # Only pass feature_names when explicitly provided (clean names matching FEATURE_TO_GROUP)
            grouped_feature_names = self.feature_names if self._explicit_feature_names else None
            explainable_predictor = ExplainableModelPredictor(feature_names=grouped_feature_names)
            explainable_results = explainable_predictor.train_and_evaluate(X_train, X_test, y_train, y_test, epochs=300, results_dir=self.results_dir)
            self.all_results.update(explainable_results)
        
        # 4. KAN模型
        if KAN_AVAILABLE:
            logging.info("Training KAN model...")
            kan_predictor = KANPredictor()
            kan_results = kan_predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
            self.all_results.update(kan_results)

        # 4b. TabNet模型
        if TABNET_AVAILABLE:
            logging.info("Training TabNet model...")
            tabnet_predictor = TabNetPredictor()
            tabnet_results = tabnet_predictor.train_and_evaluate(X_train, X_test, y_train, y_test, results_dir=self.results_dir)
            self.all_results.update(tabnet_results)

        # 5. 消融实验 (暂时禁用以避免复杂性)
        if run_ablation and include_explainable:
            logging.info("Running ablation study...")
            ablation = AblationExperiment(feature_names=self.feature_names if self._explicit_feature_names else None)
            ablation.create_ablation_models()
            ablation_results = ablation.run_ablation_study(
                X_train, X_test, y_train, y_test,
                results_dir=os.path.join(self.results_dir, 'ablation_results'),
                epochs=300
            )
        
        # 6. 高级分析
        logging.info("Running advanced model analysis...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 直接使用results_dir，让AdvancedModelAnalyzer自己创建advanced_analysis目录
        advanced_analyzer = AdvancedModelAnalyzer()
        advanced_analyzer.comprehensive_model_analysis(
            self.all_results, X_train, X_test, self.results_dir, timestamp, y_test
        )
        
        logging.info("Advanced analysis completed")
        
        # 7. 保存结果和生成报告
        logging.info("Saving results...")
        self.save_results()
        
        logging.info("Creating comprehensive comparison plots...")
        self.plot_comprehensive_comparison()
        
        logging.info("Generating model evaluation report...")
        self.generate_model_report()
        
        logging.info(f"Training completed. Total models: {len(self.all_results)}")

        # 8. 5折交叉验证
        logging.info("Running 5-fold cross-validation...")
        cv_results = self.run_cross_validation(n_splits=5)
        self.cv_results = cv_results

        # 9. 统计显著性检验
        if cv_results:
            best_cv_model = max(cv_results.keys(), key=lambda x: cv_results[x]['CV R² Mean'])
            logging.info(f"Best CV model: {best_cv_model}")
            stat_results = self.run_statistical_significance_tests(cv_results, reference_model=best_cv_model)
            self.stat_results = stat_results

        return self.all_results
        
    def save_results(self):
        """保存结果到文件"""
        logging.info("Saving results...")
        
        # 保存数值结果
        results_data = []
        for model_name, metrics in self.all_results.items():
            results_data.append({
                'Model': model_name,
                'Train_R2': metrics['Train R²'],
                'Test_R2': metrics['Test R²'],
                'Train_MSE': metrics['Train MSE'],
                'Test_MSE': metrics['Test MSE'],
                'Train_MAE': metrics['Train MAE'],
                'Test_MAE': metrics['Test MAE']
            })
        
        df_results = pd.DataFrame(results_data)
        results_file = os.path.join(self.results_dir, f'model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_results.to_csv(results_file, index=False)
        logging.info(f"Results saved to {results_file}")
        
        return df_results
        
    def plot_comprehensive_comparison(self):
        """绘制综合比较图表"""
        logging.info("Creating comprehensive comparison plots...")
        
        # 创建结果DataFrame
        metrics_data = []
        for model_name, metrics in self.all_results.items():
            metrics_data.append({
                'Model': model_name,
                'Test R²': metrics['Test R²'],
                'Test MSE': metrics['Test MSE'],
                'Test MAE': metrics['Test MAE']
            })
        
        df_results = pd.DataFrame(metrics_data)
        
        # 创建综合比较图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² 得分比较
        df_sorted = df_results.sort_values('Test R²', ascending=True)
        bars1 = axes[0, 0].barh(df_sorted['Model'], df_sorted['Test R²'])
        axes[0, 0].set_title('R² Score (Higher is Better)')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 为最佳模型着色
        max_idx = df_sorted['Test R²'].argmax()
        bars1[max_idx].set_color('gold')
        
        # MSE 比较
        df_sorted = df_results.sort_values('Test MSE', ascending=False)
        bars2 = axes[0, 1].barh(df_sorted['Model'], df_sorted['Test MSE'])
        axes[0, 1].set_title('Mean Squared Error (Lower is Better)')
        axes[0, 1].set_xlabel('MSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 为最佳模型着色
        best_mse_idx = len(df_sorted) - 1  # 最后一个是最小的
        bars2[best_mse_idx].set_color('gold')
        
        # MAE 比较
        df_sorted = df_results.sort_values('Test MAE', ascending=False)
        bars3 = axes[1, 0].barh(df_sorted['Model'], df_sorted['Test MAE'])
        axes[1, 0].set_title('Mean Absolute Error (Lower is Better)')
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 为最佳模型着色
        best_mae_idx = len(df_sorted) - 1
        bars3[best_mae_idx].set_color('gold')
        
        # 综合性能雷达图（标准化指标）
        metrics_norm = df_results.copy()
        metrics_norm['Test R²'] = (metrics_norm['Test R²'] - metrics_norm['Test R²'].min()) / (metrics_norm['Test R²'].max() - metrics_norm['Test R²'].min())
        metrics_norm['Test MSE'] = 1 - (metrics_norm['Test MSE'] - metrics_norm['Test MSE'].min()) / (metrics_norm['Test MSE'].max() - metrics_norm['Test MSE'].min())
        metrics_norm['Test MAE'] = 1 - (metrics_norm['Test MAE'] - metrics_norm['Test MAE'].min()) / (metrics_norm['Test MAE'].max() - metrics_norm['Test MAE'].min())
        
        # 综合得分
        metrics_norm['Composite Score'] = (metrics_norm['Test R²'] + metrics_norm['Test MSE'] + metrics_norm['Test MAE']) / 3
        metrics_norm = metrics_norm.sort_values('Composite Score', ascending=True)
        
        bars4 = axes[1, 1].barh(metrics_norm['Model'], metrics_norm['Composite Score'])
        axes[1, 1].set_title('Composite Performance Score')
        axes[1, 1].set_xlabel('Normalized Score (Higher is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 为最佳模型着色
        best_composite_idx = len(metrics_norm) - 1
        bars4[best_composite_idx].set_color('gold')
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.results_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logging.info(f"Comparison plot saved to {plot_file}")
        
        plt.show()
        
        return df_results
        
    def generate_model_report(self):
        """生成模型评估报告"""
        logging.info("Generating model evaluation report...")
        
        # 找到最佳模型
        best_r2_model = max(self.all_results.keys(), key=lambda x: self.all_results[x]['Test R²'])
        best_mse_model = min(self.all_results.keys(), key=lambda x: self.all_results[x]['Test MSE'])
        best_mae_model = min(self.all_results.keys(), key=lambda x: self.all_results[x]['Test MAE'])
        
        report = f"""
# DBPs预测模型综合评估报告

## 实验设置
- 数据集: {self.data_path}
- 特征数量: {len(self.feature_names)}
- 目标变量数量: {len(self.target_names)}
- 训练模型数量: {len(self.all_results)}

## 特征变量
{', '.join(self.feature_names)}

## 目标变量
{', '.join(self.target_names)}

## 模型性能总结

### 最佳R²得分模型: {best_r2_model}
- Test R²: {self.all_results[best_r2_model]['Test R²']:.4f}
- Test MSE: {self.all_results[best_r2_model]['Test MSE']:.4f}
- Test MAE: {self.all_results[best_r2_model]['Test MAE']:.4f}

### 最佳MSE模型: {best_mse_model}
- Test R²: {self.all_results[best_mse_model]['Test R²']:.4f}
- Test MSE: {self.all_results[best_mse_model]['Test MSE']:.4f}
- Test MAE: {self.all_results[best_mse_model]['Test MAE']:.4f}

### 最佳MAE模型: {best_mae_model}
- Test R²: {self.all_results[best_mae_model]['Test R²']:.4f}
- Test MSE: {self.all_results[best_mae_model]['Test MSE']:.4f}
- Test MAE: {self.all_results[best_mae_model]['Test MAE']:.4f}

## 详细结果
"""
        
        for model_name, metrics in sorted(self.all_results.items(), key=lambda x: x[1]['Test R²'], reverse=True):
            report += f"""
### {model_name}
- Train R²: {metrics['Train R²']:.4f} | Test R²: {metrics['Test R²']:.4f}
- Train MSE: {metrics['Train MSE']:.4f} | Test MSE: {metrics['Test MSE']:.4f}
- Train MAE: {metrics['Train MAE']:.4f} | Test MAE: {metrics['Test MAE']:.4f}
"""
        
        # 保存报告
        report_file = os.path.join(self.results_dir, f'model_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"Evaluation report saved to {report_file}")

        return report

    def run_cross_validation(self, n_splits=5):
        """运行5折交叉验证，返回CV均值和标准差"""
        logging.info(f"Running {n_splits}-fold cross-validation...")

        from sklearn.model_selection import KFold

        # 读取原始数据
        df = pd.read_csv(self.data_path)
        if 'Sample' in df.columns:
            df = df.drop('Sample', axis=1)

        # Use stored target/feature names (set during load_and_preprocess_data)
        X = df[self.feature_names].values
        y = df[self.target_names].values

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 只对经典ML模型做CV（速度快）
        cv_models = {
            'Linear Regression': MultiOutputRegressor(LinearRegression()),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
            'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.1)),
            'Elastic Net': MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5)),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
            'Support Vector Regression': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale')),
            'K-Nearest Neighbors': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
            'Decision Tree': MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
        }

        cv_results = {}

        for model_name, model in cv_models.items():
            fold_r2_scores = []
            fold_mse_scores = []
            fold_mae_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
                X_fold_train, X_fold_test = X_scaled[train_idx], X_scaled[test_idx]
                y_fold_train, y_fold_test = y_scaled[train_idx], y_scaled[test_idx]

                try:
                    model.fit(X_fold_train, y_fold_train)
                    y_fold_pred = model.predict(X_fold_test)

                    fold_r2_scores.append(r2_score(y_fold_test, y_fold_pred))
                    fold_mse_scores.append(mean_squared_error(y_fold_test, y_fold_pred))
                    fold_mae_scores.append(mean_absolute_error(y_fold_test, y_fold_pred))
                except Exception as e:
                    logging.warning(f"CV fold {fold_idx} failed for {model_name}: {e}")

            if fold_r2_scores:
                cv_results[model_name] = {
                    'CV R² Mean': np.mean(fold_r2_scores),
                    'CV R² Std': np.std(fold_r2_scores),
                    'CV MSE Mean': np.mean(fold_mse_scores),
                    'CV MSE Std': np.std(fold_mse_scores),
                    'CV MAE Mean': np.mean(fold_mae_scores),
                    'CV MAE Std': np.std(fold_mae_scores),
                    'CV R² Scores': fold_r2_scores
                }
                logging.info(f"{model_name}: CV R² = {np.mean(fold_r2_scores):.4f} ± {np.std(fold_r2_scores):.4f}")

        # 保存CV结果
        cv_data = []
        for model_name, metrics in cv_results.items():
            cv_data.append({
                'Model': model_name,
                'CV_R2_Mean': metrics['CV R² Mean'],
                'CV_R2_Std': metrics['CV R² Std'],
                'CV_MSE_Mean': metrics['CV MSE Mean'],
                'CV_MSE_Std': metrics['CV MSE Std'],
                'CV_MAE_Mean': metrics['CV MAE Mean'],
                'CV_MAE_Std': metrics['CV MAE Std']
            })

        df_cv = pd.DataFrame(cv_data)
        cv_file = os.path.join(self.results_dir, f'cross_validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_cv.to_csv(cv_file, index=False)
        logging.info(f"Cross-validation results saved to {cv_file}")

        # 绘制CV结果图
        self.plot_cv_results(cv_results)

        return cv_results

    def plot_cv_results(self, cv_results):
        """绘制交叉验证结果图"""
        try:
            model_names = list(cv_results.keys())
            means = [cv_results[m]['CV R² Mean'] for m in model_names]
            stds = [cv_results[m]['CV R² Std'] for m in model_names]

            sorted_idx = np.argsort(means)
            sorted_names = [model_names[i] for i in sorted_idx]
            sorted_means = [means[i] for i in sorted_idx]
            sorted_stds = [stds[i] for i in sorted_idx]

            fig, ax = plt.subplots(figsize=(10, 7))
            bars = ax.barh(sorted_names, sorted_means, xerr=sorted_stds,
                          capsize=5, alpha=0.8, color='steelblue', ecolor='black')
            bars[-1].set_color('gold')

            ax.set_xlabel('R² Score (Mean ± Std)', fontsize=12)
            ax.set_title(f'5-Fold Cross-Validation Results\n(n=65 samples, 5 folds)', fontsize=13)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            cv_plot_file = os.path.join(self.results_dir, f'cv_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(cv_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"CV comparison plot saved to {cv_plot_file}")
        except Exception as e:
            logging.error(f"Error plotting CV results: {str(e)}")

    def run_statistical_significance_tests(self, cv_results, reference_model='Neural Network'):
        """对交叉验证分数进行Wilcoxon配对检验"""
        from scipy.stats import wilcoxon

        if reference_model not in cv_results:
            logging.warning(f"Reference model '{reference_model}' not in CV results, skipping significance tests.")
            return {}

        logging.info(f"Running statistical significance tests (reference: {reference_model})...")

        ref_scores = cv_results[reference_model]['CV R² Scores']
        test_results = {}

        for model_name, metrics in cv_results.items():
            if model_name == reference_model:
                continue
            scores = metrics['CV R² Scores']
            if len(scores) < 2:
                continue
            try:
                stat, p_value = wilcoxon(ref_scores, scores)
                test_results[model_name] = {
                    'W_statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'ref_mean': np.mean(ref_scores),
                    'model_mean': np.mean(scores),
                    'delta_R2': np.mean(ref_scores) - np.mean(scores)
                }
                sig_mark = '*' if p_value < 0.05 else 'ns'
                logging.info(f"  {reference_model} vs {model_name}: p={p_value:.4f} ({sig_mark}), ΔR²={np.mean(ref_scores)-np.mean(scores):.4f}")
            except Exception as e:
                logging.warning(f"  Wilcoxon test failed for {model_name}: {e}")

        # 保存统计检验结果
        stat_data = []
        for model_name, res in test_results.items():
            stat_data.append({
                'Model': model_name,
                'Reference': reference_model,
                'W_statistic': res['W_statistic'],
                'p_value': res['p_value'],
                'Significant (p<0.05)': res['significant'],
                'Reference_R2_Mean': res['ref_mean'],
                'Model_R2_Mean': res['model_mean'],
                'Delta_R2': res['delta_R2']
            })

        if stat_data:
            df_stat = pd.DataFrame(stat_data)
            stat_file = os.path.join(self.results_dir, f'statistical_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df_stat.to_csv(stat_file, index=False)
            logging.info(f"Statistical test results saved to {stat_file}")

        return test_results


class SimpleExplainableModel(torch.nn.Module):
    """简化的可解释模型，用于消融研究"""
    
    def __init__(self, input_dim, hidden_dim=64, num_targets=5):
        super(SimpleExplainableModel, self).__init__()
        
        # 简单的特征提取层
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        # 注意力机制
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, input_dim),
            torch.nn.Softmax(dim=1)
        )
        
        # 预测器
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_targets)
        )
        
    def forward(self, x, return_attention=False):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 计算注意力权重
        attention_weights = self.attention(features)
        
        # 应用注意力权重到原始输入
        attended_input = x * attention_weights
        
        # 重新提取特征
        final_features = self.feature_extractor(attended_input)
        
        # 预测
        predictions = self.predictor(final_features)
        
        if return_attention:
            return predictions, attention_weights
        return predictions

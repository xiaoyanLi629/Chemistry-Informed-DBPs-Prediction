#!/usr/bin/env python3
"""
Multi-seed experiment runner.
Runs all models with multiple random seeds and reports mean ± std.
Supports both original and Ontario datasets.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
from models import (ExplainableModelPredictor, NeuralNetworkPredictor, KANPredictor)

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("Warning: pytorch_tabnet not available, TabNet will be skipped.")

# ── Configuration ───────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
TEST_SIZE = 0.3

DATASETS = {
    'original': {
        'data_path': 'data/data.csv',
        'target_names': None,  # auto-detect (first 5 cols)
        'feature_names': None,  # auto-detect (remaining cols)
        'results_dir': 'results_legacy/conventional_protocol/original',
    },
    'ontario': {
        'data_path': 'data/ontario_data.csv',
        'target_names': ['DCAA', 'TCAA', 'HAA5', 'TTHM'],
        'feature_names': ['Temp', 'pH', 'DOC', 'Cl2', 'NO2-N', 'NH4-N', 'Br'],
        'results_dir': 'results_legacy/conventional_protocol/ontario',
    },
}

# sklearn models
def get_sklearn_models():
    return {
        'Linear Regression': MultiOutputRegressor(LinearRegression()),
        'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
        'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.1, max_iter=10000)),
        'Elastic Net': MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        'Extra Trees': MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, random_state=42)),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        'Support Vector Regression': MultiOutputRegressor(SVR(kernel='rbf', C=1.0)),
        'K-Nearest Neighbors': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
        'Decision Tree': MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
    }


def train_tabnet(X_train, X_test, y_train, y_test, seed):
    """Train TabNet model."""
    if not TABNET_AVAILABLE:
        return None
    model = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3,
        gamma=1.3, lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        seed=seed,
        verbose=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=50,
        batch_size=16,
    )
    y_pred = model.predict(X_test)
    return {
        'Train R²': r2_score(y_train, model.predict(X_train)),
        'Test R²': r2_score(y_test, y_pred),
        'Train MSE': mean_squared_error(y_train, model.predict(X_train)),
        'Test MSE': mean_squared_error(y_test, y_pred),
        'Train MAE': mean_absolute_error(y_train, model.predict(X_train)),
        'Test MAE': mean_absolute_error(y_test, y_pred),
    }


def train_neural_net(X_train, X_test, y_train, y_test, seed):
    """Train vanilla Neural Network."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    predictor = NeuralNetworkPredictor()
    results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test, epochs=300)
    if 'Neural Network' in results:
        return results['Neural Network']
    return None


def train_explainable(X_train, X_test, y_train, y_test, seed, feature_names=None,
                      target_names=None, x_mean=None, x_std=None, y_mean=None, y_std=None):
    """Train Explainable Model with chemistry-informed features and constraints."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    predictor = ExplainableModelPredictor(feature_names=feature_names, target_names=target_names)
    results = predictor.train_and_evaluate(
        X_train, X_test, y_train, y_test, epochs=300,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std
    )
    if 'Explainable Model' in results:
        return results['Explainable Model']
    return None


def train_kan(X_train, X_test, y_train, y_test, seed):
    """Train KAN model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    predictor = KANPredictor()
    results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    if 'KAN Model' in results:
        return results['KAN Model']
    return None


def run_single_seed(X, y, seed, feature_names=None, target_names=None):
    """Run all models for a single train/test split."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )

    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    results = {}

    # 1. Sklearn models
    for name, model in get_sklearn_models().items():
        try:
            model.fit(X_train_s, y_train_s)
            y_pred_train = model.predict(X_train_s)
            y_pred_test = model.predict(X_test_s)
            results[name] = {
                'Train R²': r2_score(y_train_s, y_pred_train),
                'Test R²': r2_score(y_test_s, y_pred_test),
                'Train MSE': mean_squared_error(y_train_s, y_pred_train),
                'Test MSE': mean_squared_error(y_test_s, y_pred_test),
                'Train MAE': mean_absolute_error(y_train_s, y_pred_train),
                'Test MAE': mean_absolute_error(y_test_s, y_pred_test),
            }
        except Exception as e:
            print(f"  Error training {name}: {e}")

    # 2. TabNet
    try:
        r = train_tabnet(X_train_s, X_test_s, y_train_s, y_test_s, seed)
        if r:
            results['TabNet'] = r
    except Exception as e:
        print(f"  Error training TabNet: {e}")

    # 3. Neural Network
    try:
        r = train_neural_net(X_train_s, X_test_s, y_train_s, y_test_s, seed)
        if r:
            results['Neural Network'] = r
    except Exception as e:
        print(f"  Error training Neural Network: {e}")

    # 4. Explainable Model (with chemistry-informed features and hierarchical loss)
    try:
        r = train_explainable(X_train_s, X_test_s, y_train_s, y_test_s, seed,
                              feature_names=feature_names, target_names=target_names,
                              x_mean=scaler_X.mean_, x_std=scaler_X.scale_,
                              y_mean=scaler_y.mean_, y_std=scaler_y.scale_)
        if r:
            results['Explainable Model'] = r
    except Exception as e:
        print(f"  Error training Explainable Model: {e}")

    # 5. KAN
    try:
        r = train_kan(X_train_s, X_test_s, y_train_s, y_test_s, seed)
        if r:
            results['KAN Model'] = r
    except Exception as e:
        print(f"  Error training KAN: {e}")

    return results


def run_dataset(dataset_name, config):
    """Run multi-seed experiments for one dataset."""
    print(f"\n{'='*80}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*80}")

    # Load data
    df = pd.read_csv(config['data_path'])
    if 'Sample' in df.columns:
        df = df.drop('Sample', axis=1)

    if config['target_names'] and config['feature_names']:
        target_names = config['target_names']
        feature_names = config['feature_names']
        X = df[feature_names].values
        y = df[target_names].values
    else:
        target_names = df.columns[:5].tolist()
        feature_names = df.columns[5:].tolist()
        X = df.iloc[:, 5:].values
        y = df.iloc[:, :5].values

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Targets: {y.shape[1]}")
    print(f"  Features: {feature_names}")
    print(f"  Targets: {target_names}")
    print(f"  Seeds: {SEEDS}")

    # Run all seeds
    all_seed_results = {}
    for i, seed in enumerate(SEEDS):
        print(f"\n  --- Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        t0 = time.time()
        seed_results = run_single_seed(X, y, seed, feature_names=feature_names, target_names=target_names)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s, {len(seed_results)} models")
        all_seed_results[seed] = seed_results

    # Aggregate results: mean ± std for each metric
    model_names = set()
    for sr in all_seed_results.values():
        model_names.update(sr.keys())
    model_names = sorted(model_names)

    metrics = ['Train R²', 'Test R²', 'Train MSE', 'Test MSE', 'Train MAE', 'Test MAE']
    summary_rows = []

    for model in model_names:
        row = {'Model': model}
        values = {m: [] for m in metrics}
        for seed, sr in all_seed_results.items():
            if model in sr:
                for m in metrics:
                    if m in sr[model]:
                        values[m].append(sr[model][m])
        for m in metrics:
            if values[m]:
                row[f'{m} Mean'] = np.mean(values[m])
                row[f'{m} Std'] = np.std(values[m])
                row[f'{m} Values'] = values[m]
            else:
                row[f'{m} Mean'] = np.nan
                row[f'{m} Std'] = np.nan
        row['N_seeds'] = len([s for s in all_seed_results if model in all_seed_results[s]])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Test R² Mean', ascending=False).reset_index(drop=True)

    # Save results
    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Save detailed CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_cols = ['Model', 'N_seeds']
    for m in metrics:
        csv_cols.extend([f'{m} Mean', f'{m} Std'])
    summary_df[csv_cols].to_csv(
        os.path.join(results_dir, f'multiseed_results_{ts}.csv'), index=False
    )

    # Save per-seed raw results
    raw_rows = []
    for seed, sr in all_seed_results.items():
        for model, vals in sr.items():
            raw_row = {'Seed': seed, 'Model': model}
            raw_row.update(vals)
            raw_rows.append(raw_row)
    pd.DataFrame(raw_rows).to_csv(
        os.path.join(results_dir, f'per_seed_results_{ts}.csv'), index=False
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Results: {dataset_name} ({len(SEEDS)} seeds)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Model':<28} {'Test R² (mean±std)':<22} {'Test MSE (mean±std)':<22} {'Test MAE (mean±std)':<22}")
    print("-" * 100)

    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        r2_str = f"{row['Test R² Mean']:.3f}±{row['Test R² Std']:.3f}"
        mse_str = f"{row['Test MSE Mean']:.3f}±{row['Test MSE Std']:.3f}"
        mae_str = f"{row['Test MAE Mean']:.3f}±{row['Test MAE Std']:.3f}"
        print(f"{rank:<5} {row['Model']:<28} {r2_str:<22} {mse_str:<22} {mae_str:<22}")

    # Also save the summary as a clean CSV for the paper
    paper_rows = []
    for _, row in summary_df.iterrows():
        paper_rows.append({
            'Model': row['Model'],
            'Train_R2': row['Train R² Mean'],
            'Train_R2_std': row['Train R² Std'],
            'Test_R2': row['Test R² Mean'],
            'Test_R2_std': row['Test R² Std'],
            'Train_MSE': row['Train MSE Mean'],
            'Train_MSE_std': row['Train MSE Std'],
            'Test_MSE': row['Test MSE Mean'],
            'Test_MSE_std': row['Test MSE Std'],
            'Train_MAE': row['Train MAE Mean'],
            'Train_MAE_std': row['Train MAE Std'],
            'Test_MAE': row['Test MAE Mean'],
            'Test_MAE_std': row['Test MAE Std'],
        })
    pd.DataFrame(paper_rows).to_csv(
        os.path.join(results_dir, 'model_results.csv'), index=False
    )

    return summary_df


if __name__ == '__main__':
    print("=" * 80)
    print("  Multi-Seed Experiment Runner")
    print(f"  Seeds: {SEEDS}")
    print(f"  Test size: {TEST_SIZE}")
    print("=" * 80)

    t_start = time.time()

    results = {}
    for ds_name, ds_config in DATASETS.items():
        if os.path.exists(ds_config['data_path']):
            results[ds_name] = run_dataset(ds_name, ds_config)
        else:
            print(f"\nSkipping {ds_name}: {ds_config['data_path']} not found")

    total = time.time() - t_start
    print(f"\n\nTotal time: {total:.1f}s ({total/60:.1f}min)")

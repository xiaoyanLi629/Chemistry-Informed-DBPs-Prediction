#!/usr/bin/env python3
"""Leak-free multi-seed benchmark .

Strict evaluation protocol:
- validation set carved from the training split; early stopping, LR scheduling
  and TabNet model selection all use validation data only;
- all sklearn baselines hyperparameter-tuned with inner CV on the train split;
- adds XGBoost / LightGBM / CatBoost, chemistry-feature-augmented simple
  baselines, and a parameter-comparable multi-task MLP;
- reports per-target R2 and trainable-parameter counts;
- 10 random seeds.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from experiments.baselines import SKLEARN_SPECS, add_chemistry_features, fit_tuned
from experiments.nets import MultiTaskMLP
from experiments.protocol import (SEEDS_10, count_parameters, evaluate, scale_all,
                                  split_train_val_test, torch_predict,
                                  train_torch_model)
from models import ExplainableDBPsModel, HierarchicalConsistencyLoss

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

try:
    from kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False

DATASETS = {
    'original': {
        'data_path': 'data/data.csv',
        'target_names': None,
        'feature_names': None,
    },
    'ontario': {
        'data_path': 'data/ontario_data.csv',
        'target_names': ['DCAA', 'TCAA', 'HAA5', 'TTHM'],
        'feature_names': ['Temp', 'pH', 'DOC', 'Cl2', 'NO2-N', 'NH4-N', 'Br'],
    },
}

CHEM_AUGMENTED = ['Support Vector Regression', 'Random Forest', 'Ridge Regression']


def load_dataset(name):
    cfg = DATASETS[name]
    df = pd.read_csv(cfg['data_path'])
    if 'Sample' in df.columns:
        df = df.drop('Sample', axis=1)
    if cfg['target_names'] and cfg['feature_names']:
        target_names, feature_names = cfg['target_names'], cfg['feature_names']
        X = df[feature_names].values.astype(float)
        y = df[target_names].values.astype(float)
    else:
        target_names = df.columns[:5].tolist()
        feature_names = df.columns[5:].tolist()
        X = df.iloc[:, 5:].values.astype(float)
        y = df.iloc[:, :5].values.astype(float)
    return X, y, feature_names, target_names


def _resolve_splits(X, y, seed, splits, val_frac=0.2):
    """Return raw (unscaled) train/val/test arrays plus index bookkeeping."""
    if splits is None:
        return split_train_val_test(X, y, seed, test_size=0.3, val_size=val_frac)
    if len(splits) == 3:
        tr_idx, val_idx, te_idx = splits
        return (X[tr_idx], X[val_idx], X[te_idx], y[tr_idx], y[val_idx], y[te_idx])
    tr_idx, te_idx = splits
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr_idx))
    n_val = max(1, int(round(len(tr_idx) * val_frac)))
    val_idx = np.asarray(tr_idx)[perm[:n_val]]
    tr_only = np.asarray(tr_idx)[perm[n_val:]]
    return (X[tr_only], X[val_idx], X[te_idx], y[tr_only], y[val_idx], y[te_idx])


def _metrics_row(predict_fn, d, target_names, params=np.nan):
    row = {}
    tr = evaluate(predict_fn, d['X_tr'], d['y_tr'], target_names)
    va = evaluate(predict_fn, d['X_val'], d['y_val'], target_names)
    te = evaluate(predict_fn, d['X_te'], d['y_te'], target_names)
    row['Train R2'] = tr['R2']
    row['Val R2'] = va['R2']
    row['Test R2'] = te['R2']
    row['Test MSE'] = te['MSE']
    row['Test MAE'] = te['MAE']
    for t in target_names:
        row[f'Test R2 {t}'] = te[f'R2_{t}']
    row['Params'] = params
    return row


def train_explainable_model(d, feature_names, target_names, seed,
                            lambda_h=0.1, use_bcaa=True, nonneg=True,
                            epochs=1000, patience=100, **variant_flags):
    """Train the explainable model under the leak-free protocol.

    Returns (metrics_row, model, hier_loss_fn). Reused by ablation/lambda tasks.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ExplainableDBPsModel(
        input_dim=d['X_tr'].shape[1], num_targets=d['y_tr'].shape[1],
        feature_names=feature_names,
        x_mean=d['scaler_X'].mean_, x_std=d['scaler_X'].scale_,
        **variant_flags)
    hier = HierarchicalConsistencyLoss(
        target_names, y_mean=d['scaler_y'].mean_, y_std=d['scaler_y'].scale_,
        lambda_h=lambda_h, use_bcaa=use_bcaa, nonneg=nonneg)
    mse = torch.nn.MSELoss()

    def loss_fn(pred, target):
        return mse(pred, target) + hier(pred)

    train_torch_model(model, loss_fn, d['X_tr'], d['y_tr'], d['X_val'], d['y_val'],
                      epochs=epochs, patience=patience)
    row = _metrics_row(lambda X_: torch_predict(model, X_), d, target_names,
                       params=count_parameters(model))
    return row, model, hier


def train_mlp(d, target_names, seed, hidden_dim=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiTaskMLP(d['X_tr'].shape[1], d['y_tr'].shape[1], hidden_dim=hidden_dim)
    train_torch_model(model, torch.nn.MSELoss(), d['X_tr'], d['y_tr'],
                      d['X_val'], d['y_val'], epochs=1000, patience=100)
    return _metrics_row(lambda X_: torch_predict(model, X_), d, target_names,
                        params=count_parameters(model))


def train_tabnet_leakfree(d, target_names, seed):
    if not TABNET_AVAILABLE:
        return None
    model = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax', seed=seed, verbose=0)
    model.fit(d['X_tr'], d['y_tr'],
              eval_set=[(d['X_val'], d['y_val'])],  # validation, not test
              eval_metric=['rmse'], max_epochs=200, patience=50, batch_size=16)
    n_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    return _metrics_row(model.predict, d, target_names, params=n_params)


def train_kan_leakfree(d, target_names, seed):
    if not KAN_AVAILABLE:
        return None
    torch.manual_seed(seed)
    np.random.seed(seed)
    import tempfile
    ckpt_dir = tempfile.mkdtemp(prefix='kan_ckpt_')
    model = KAN(width=[d['X_tr'].shape[1], 3, d['y_tr'].shape[1]], grid=5, k=3, seed=seed,
                ckpt_path=ckpt_dir)
    dataset = {
        'train_input': torch.FloatTensor(d['X_tr']),
        'train_label': torch.FloatTensor(d['y_tr']),
        # KAN.fit only logs on these; fixed step count, no model selection
        'test_input': torch.FloatTensor(d['X_val']),
        'test_label': torch.FloatTensor(d['y_val']),
    }
    model.fit(dataset, opt="LBFGS", steps=50, lamb=0.01)
    model = model.prune()
    model.fit(dataset, opt="LBFGS", steps=50, lr=0.001, lamb=0.1)

    def predict(X_):
        with torch.no_grad():
            return model(torch.FloatTensor(X_)).numpy()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _metrics_row(predict, d, target_names, params=n_params)


def run_one_seed(X, y, feature_names, target_names, seed, models='all', splits=None,
                 tune=True):
    """Run the requested model set for one train/val/test split. X and y are raw."""
    parts = _resolve_splits(X, y, seed, splits)
    d = scale_all(*parts)
    results = {}

    wanted = None if models == 'all' else set(models)

    def want(name):
        return wanted is None or name in wanted

    # sklearn baselines (tuned)
    for name in SKLEARN_SPECS:
        if not want(name):
            continue
        try:
            m = fit_tuned(name, d['X_tr'], d['y_tr'], seed, tune=tune)
            results[name] = _metrics_row(m.predict, d, target_names)
        except Exception as e:
            print(f"  [seed {seed}] {name} failed: {e}")

    # chemistry-feature-augmented simple baselines (raw-scale augmentation)
    X_aug, _aug_names = add_chemistry_features(X, feature_names)
    if X_aug.shape[1] > X.shape[1]:
        parts_aug = _resolve_splits(X_aug, y, seed, splits)
        d_aug = scale_all(*parts_aug)
        for name in CHEM_AUGMENTED:
            label = f"{name}+chem"
            if not want(label):
                continue
            try:
                m = fit_tuned(name, d_aug['X_tr'], d_aug['y_tr'], seed, tune=tune)
                results[label] = _metrics_row(m.predict, d_aug, target_names)
            except Exception as e:
                print(f"  [seed {seed}] {label} failed: {e}")

    # TabNet
    if want('TabNet'):
        try:
            r = train_tabnet_leakfree(d, target_names, seed)
            if r:
                results['TabNet'] = r
        except Exception as e:
            print(f"  [seed {seed}] TabNet failed: {e}")

    # multi-task MLP
    if want('Multi-task MLP'):
        try:
            results['Multi-task MLP'] = train_mlp(d, target_names, seed)
        except Exception as e:
            print(f"  [seed {seed}] MLP failed: {e}")

    # Explainable model
    if want('Explainable Model'):
        try:
            row, _, _ = train_explainable_model(d, feature_names, target_names, seed)
            results['Explainable Model'] = row
        except Exception as e:
            print(f"  [seed {seed}] Explainable Model failed: {e}")

    # KAN
    if want('KAN Model'):
        try:
            r = train_kan_leakfree(d, target_names, seed)
            if r:
                results['KAN Model'] = r
        except Exception as e:
            print(f"  [seed {seed}] KAN failed: {e}")

    return results


def summarize(per_seed_df, target_names):
    metrics = (['Train R2', 'Val R2', 'Test R2', 'Test MSE', 'Test MAE'] +
               [f'Test R2 {t}' for t in target_names])
    rows = []
    for model, g in per_seed_df.groupby('Model'):
        row = {'Model': model, 'N_seeds': len(g), 'Params': g['Params'].iloc[0]}
        for m in metrics:
            row[f'{m} Mean'] = g[m].mean()
            row[f'{m} Std'] = g[m].std(ddof=0)
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values('Test R2 Mean', ascending=False)
            .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='both', choices=['original', 'ontario', 'both'])
    ap.add_argument('--seeds', default=None,
                    help='comma-separated seed list (default: the 10 standard seeds)')
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-tune', action='store_true')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS_10
    ds_names = ['original', 'ontario'] if args.dataset == 'both' else [args.dataset]
    os.makedirs(args.out, exist_ok=True)

    for ds in ds_names:
        X, y, feature_names, target_names = load_dataset(ds)
        print(f"\n=== {ds}: n={len(X)}, {len(feature_names)} features, "
              f"{len(target_names)} targets, seeds={seeds} ===")
        rows = []
        for i, seed in enumerate(seeds):
            t0 = time.time()
            res = run_one_seed(X, y, feature_names, target_names, seed,
                               tune=not args.no_tune)
            for model, r in res.items():
                rows.append({'Dataset': ds, 'Seed': seed, 'Model': model, **r})
            print(f"  seed {seed} ({i+1}/{len(seeds)}): {len(res)} models, "
                  f"{time.time()-t0:.1f}s")
        per_seed = pd.DataFrame(rows)
        per_seed.to_csv(os.path.join(args.out, f'benchmark_{ds}_per_seed.csv'), index=False)
        summary = summarize(per_seed, target_names)
        summary.to_csv(os.path.join(args.out, f'benchmark_{ds}_summary.csv'), index=False)

        print(f"\n  {'Model':<32} {'Test R2':<16} {'Params'}")
        for _, r in summary.iterrows():
            print(f"  {r['Model']:<32} {r['Test R2 Mean']:.3f}±{r['Test R2 Std']:.3f}"
                  f"   {'' if pd.isna(r['Params']) else int(r['Params'])}")


if __name__ == '__main__':
    main()

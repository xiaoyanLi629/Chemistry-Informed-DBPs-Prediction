#!/usr/bin/env python3
"""Regularization and capacity sweep for the explainable model (v2 protocol).

One-factor-at-a-time sweep around the default configuration, plus one
"strong regularization" arm combining the three levers, over 10 seeds and both
datasets. Reports train / validation / test R2 and the train-test gap so the
effect of each lever on overfitting is visible, not only its effect on
accuracy, so the effect of dropout, weight decay and reduced capacity on the
small-data overfitting gap can be assessed directly.

Levers:
- weight decay:  1e-5 (default), 1e-4, 1e-3
- dropout:       default (0.1 extractors / 0.2 heads), 0.3 (all), 0.5 (all)
- capacity:      hidden 32 + heads 128/64 (default), hidden 16 + heads 64/32,
                 hidden 8 + heads 32/16
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from experiments.protocol import (SEEDS_10, count_parameters, scale_all,
                                  split_train_val_test, torch_predict,
                                  train_torch_model)
from experiments.run_benchmark import _metrics_row, load_dataset
from models import ExplainableDBPsModel, HierarchicalConsistencyLoss

# arm -> (hidden_dim, head_dims, dropout_all, weight_decay)
ARMS = {
    'default':          (32, (128, 64), None, 1e-5),
    'wd_1e-4':          (32, (128, 64), None, 1e-4),
    'wd_1e-3':          (32, (128, 64), None, 1e-3),
    'dropout_0.3':      (32, (128, 64), 0.3,  1e-5),
    'dropout_0.5':      (32, (128, 64), 0.5,  1e-5),
    'capacity_16':      (16, (64, 32),  None, 1e-5),
    'capacity_8':       (8,  (32, 16),  None, 1e-5),
    'strong_reg':       (16, (64, 32),  0.3,  1e-4),
}


class SizedPredictor(torch.nn.Module):
    """Per-target heads with configurable widths (default model uses 128/64)."""

    def __init__(self, input_dim, num_targets, h1, h2, dropout=0.2):
        super().__init__()
        self.predictors = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, h1), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(h1, h2), torch.nn.ReLU(),
                torch.nn.Linear(h2, 1))
            for _ in range(num_targets)])

    def forward(self, x):
        return torch.cat([p(x) for p in self.predictors], dim=1)


def build_model(d, feature_names, hidden_dim, head_dims, dropout_all):
    model = ExplainableDBPsModel(
        input_dim=d['X_tr'].shape[1], num_targets=d['y_tr'].shape[1],
        hidden_dim=hidden_dim, feature_names=feature_names,
        x_mean=d['scaler_X'].mean_, x_std=d['scaler_X'].scale_)
    if head_dims != (128, 64):
        in_dim = model.predictor.predictors[0][0].in_features
        model.predictor = SizedPredictor(in_dim, d['y_tr'].shape[1], *head_dims)
    if dropout_all is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout_all
    return model


def run_arm(d, feature_names, target_names, seed, arm):
    hidden_dim, head_dims, dropout_all, wd = ARMS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(d, feature_names, hidden_dim, head_dims, dropout_all)
    hier = HierarchicalConsistencyLoss(
        target_names, y_mean=d['scaler_y'].mean_, y_std=d['scaler_y'].scale_,
        lambda_h=0.1, use_bcaa=True, nonneg=True)
    mse = torch.nn.MSELoss()
    hist = train_torch_model(model, lambda p, t: mse(p, t) + hier(p),
                             d['X_tr'], d['y_tr'], d['X_val'], d['y_val'],
                             epochs=1000, patience=100, weight_decay=wd)
    row = _metrics_row(lambda X_: torch_predict(model, X_), d, target_names,
                       params=count_parameters(model))
    row['Best epoch'] = hist['best_epoch']
    row['Train-Test gap'] = row['Train R2'] - row['Test R2']
    return row


def summarize(df, out_dir):
    rows = []
    for ds, g in df.groupby('Dataset'):
        base = g[g['Arm'] == 'default'].set_index('Seed')['Test R2']
        for arm in ARMS:
            ga = g[g['Arm'] == arm]
            if ga.empty:
                continue
            row = {'Dataset': ds, 'Arm': arm, 'N_seeds': len(ga),
                   'Params': int(ga['Params'].iloc[0])}
            for m in ['Train R2', 'Val R2', 'Test R2', 'Train-Test gap', 'Best epoch']:
                row[f'{m} Mean'] = ga[m].mean()
                row[f'{m} Std'] = ga[m].std(ddof=0)
            if arm != 'default':
                paired = ga.set_index('Seed')['Test R2'].reindex(base.index).dropna()
                row['Delta Test R2 vs default'] = paired.mean() - base.loc[paired.index].mean()
                if len(paired) >= 3:
                    row['p_value_paired_t'] = stats.ttest_rel(
                        paired, base.loc[paired.index]).pvalue
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'regularization_sweep_summary.csv'), index=False)
    show = ['Dataset', 'Arm', 'Params', 'Train R2 Mean', 'Test R2 Mean', 'Test R2 Std',
            'Train-Test gap Mean', 'Delta Test R2 vs default', 'p_value_paired_t']
    print(summary[show].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--datasets', default='original,ontario')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = SEEDS_10[:2] if args.quick else SEEDS_10

    rows = []
    for ds in args.datasets.split(','):
        X, y, feature_names, target_names = load_dataset(ds)
        for seed in seeds:
            d = scale_all(*split_train_val_test(X, y, seed))
            for arm in ARMS:
                t0 = time.time()
                row = run_arm(d, feature_names, target_names, seed, arm)
                rows.append({'Dataset': ds, 'Seed': seed, 'Arm': arm, **row})
                print(f"{ds} seed {seed} {arm:<14} params={int(row['Params']):>7} "
                      f"train={row['Train R2']:.3f} test={row['Test R2']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'regularization_sweep_per_seed.csv'), index=False)
    summarize(df, args.out)


if __name__ == '__main__':
    main()

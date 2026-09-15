#!/usr/bin/env python3
"""Hierarchical consistency loss applied to the unstructured multi-task MLP.

The hierarchical loss is not architecture-specific, so a fair comparison must
give the baseline the same loss. This trains the multi-task MLP with and
without the hierarchical penalty (lambda_h = 0.1, identical to the explainable
model) over 10 seeds on both datasets, with paired t-tests and constraint
violation rates. Also re-evaluates the full explainable model in the same
loop so the three arms share identical splits.
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

from experiments.nets import MultiTaskMLP
from experiments.protocol import (SEEDS_10, count_parameters, scale_all,
                                  split_train_val_test, torch_predict,
                                  train_torch_model)
from experiments.run_benchmark import (_metrics_row, load_dataset,
                                       train_explainable_model)
from models import HierarchicalConsistencyLoss


def train_mlp_variant(d, target_names, seed, lambda_h, hidden_dim=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiTaskMLP(d['X_tr'].shape[1], d['y_tr'].shape[1], hidden_dim=hidden_dim)
    hier = HierarchicalConsistencyLoss(
        target_names, y_mean=d['scaler_y'].mean_, y_std=d['scaler_y'].scale_,
        lambda_h=lambda_h, use_bcaa=True, nonneg=True)
    mse = torch.nn.MSELoss()
    train_torch_model(model, lambda p, t: mse(p, t) + hier(p),
                      d['X_tr'], d['y_tr'], d['X_val'], d['y_val'],
                      epochs=1000, patience=100)
    row = _metrics_row(lambda X_: torch_predict(model, X_), d, target_names,
                       params=count_parameters(model))
    pred_te = torch.FloatTensor(torch_predict(model, d['X_te']))
    row.update(hier.violation_stats(pred_te))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = SEEDS_10[:2] if args.quick else SEEDS_10

    rows = []
    for ds in ['original', 'ontario']:
        X, y, feature_names, target_names = load_dataset(ds)
        for seed in seeds:
            d = scale_all(*split_train_val_test(X, y, seed))
            t0 = time.time()
            r = train_mlp_variant(d, target_names, seed, lambda_h=0.0)
            rows.append({'Dataset': ds, 'Seed': seed, 'Arm': 'mlp', **r})
            r = train_mlp_variant(d, target_names, seed, lambda_h=0.1)
            rows.append({'Dataset': ds, 'Seed': seed, 'Arm': 'mlp_hier', **r})
            r, model, hier = train_explainable_model(d, feature_names, target_names, seed)
            r.update(hier.violation_stats(torch.FloatTensor(torch_predict(model, d['X_te']))))
            rows.append({'Dataset': ds, 'Seed': seed, 'Arm': 'explainable', **r})
            print(f"{ds} seed {seed}: mlp={rows[-3]['Test R2']:.3f} "
                  f"mlp_hier={rows[-2]['Test R2']:.3f} "
                  f"explainable={rows[-1]['Test R2']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'mlp_hier_per_seed.csv'), index=False)

    out = []
    viol_cols = [c for c in df.columns if c.startswith('viol_')]
    for ds, g in df.groupby('Dataset'):
        piv = g.pivot(index='Seed', columns='Arm', values='Test R2')
        for arm in ['mlp', 'mlp_hier', 'explainable']:
            ga = g[g['Arm'] == arm]
            row = {'Dataset': ds, 'Arm': arm, 'N_seeds': len(ga),
                   'Params': int(ga['Params'].iloc[0]),
                   'Train R2 Mean': ga['Train R2'].mean(),
                   'Test R2 Mean': ga['Test R2'].mean(),
                   'Test R2 Std': ga['Test R2'].std(ddof=0)}
            for c in viol_cols:
                row[f'{c} Mean'] = ga[c].mean()
            if arm != 'mlp' and 'mlp' in piv:
                row['Delta vs mlp'] = (piv[arm] - piv['mlp']).mean()
                if len(piv) >= 3:
                    row['p_vs_mlp'] = stats.ttest_rel(piv[arm], piv['mlp']).pvalue
            if arm == 'explainable' and 'mlp_hier' in piv:
                row['Delta vs mlp_hier'] = (piv[arm] - piv['mlp_hier']).mean()
                if len(piv) >= 3:
                    row['p_vs_mlp_hier'] = stats.ttest_rel(piv[arm], piv['mlp_hier']).pvalue
            out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(args.out, 'mlp_hier_summary.csv'), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

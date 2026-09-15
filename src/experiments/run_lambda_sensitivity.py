#!/usr/bin/env python3
"""Hierarchical-loss lambda sensitivity and constraint-violation analysis .

Sweeps the loss weight lambda_h over 10 seeds and both datasets, reporting
overall and per-target test R2 plus the fraction of test predictions violating
each chemical constraint (with/without the penalty).

Active constraints per dataset:
- original (DCAA, TCAA, BCAA, HAA5, HAA9): HAA5 >= DCAA+TCAA+BCAA,
  HAA9 >= HAA5, non-negativity;
- ontario  (DCAA, TCAA, HAA5, TTHM):       HAA5 >= DCAA+TCAA, non-negativity.
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

from experiments.protocol import scale_all, split_train_val_test, SEEDS_10, torch_predict
from experiments.run_benchmark import load_dataset, train_explainable_model

LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = SEEDS_10[:1] if args.quick else SEEDS_10
    lambdas = [0.0, 0.1] if args.quick else LAMBDAS

    rows = []
    for ds in ['original', 'ontario']:
        X, y, feature_names, target_names = load_dataset(ds)
        for seed in seeds:
            parts = split_train_val_test(X, y, seed)
            d = scale_all(*parts)
            for lam in lambdas:
                t0 = time.time()
                row, model, hier = train_explainable_model(
                    d, feature_names, target_names, seed, lambda_h=lam)
                # violation stats on test predictions (original scale via hier's scaler)
                pred_te = torch.FloatTensor(torch_predict(model, d['X_te']))
                stats = hier.violation_stats(pred_te)
                rows.append({'Dataset': ds, 'Seed': seed, 'Lambda': lam,
                             **row, **stats})
                print(f"{ds} seed {seed} lambda {lam}: R2={row['Test R2']:.3f} "
                      f"viol={stats} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'lambda_sensitivity_per_seed.csv'), index=False)

    metric_cols = [c for c in df.columns
                   if c.startswith('Test R2') or c.startswith('viol_')]
    summary = df.groupby(['Dataset', 'Lambda'])[metric_cols].agg(['mean', 'std'])
    summary.columns = [' '.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out, 'lambda_sensitivity_summary.csv'), index=False)

    show = ['Dataset', 'Lambda', 'Test R2 mean', 'Test R2 std'] + \
           [c for c in summary.columns if c.startswith('viol_') and c.endswith('mean')]
    print(summary[show].round(4).to_string(index=False))


if __name__ == '__main__':
    main()

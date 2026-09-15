#!/usr/bin/env python3
"""Sensitivity of results to the feature/target standardization choice.

Compares z-score (StandardScaler, the default), RobustScaler (median / IQR,
less sensitive to outliers at small n) and MinMaxScaler for the explainable
model and tuned SVR over 10 seeds on both datasets. Scalers are always fit on
the training split only. R2 is invariant to affine target rescaling, so the
test R2 values are directly comparable across scalers.

The explainable model recovers raw-scale inputs internally via
x_raw = x_scaled * x_std + x_mean (for its chemistry-derived features), and
the hierarchical loss does the same for targets, so each scaler is mapped to
its equivalent affine (mean, std) parameters.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from experiments.baselines import fit_tuned
from experiments.protocol import (SEEDS_10, count_parameters, split_train_val_test,
                                  torch_predict, train_torch_model)
from experiments.run_benchmark import _metrics_row, load_dataset
from models import ExplainableDBPsModel, HierarchicalConsistencyLoss

SCALERS = {
    'zscore': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler,
}


def affine_params(scaler):
    """Return (mean, std) such that x_raw = x_scaled * std + mean."""
    if isinstance(scaler, StandardScaler):
        return scaler.mean_, scaler.scale_
    if isinstance(scaler, RobustScaler):
        return scaler.center_, scaler.scale_
    if isinstance(scaler, MinMaxScaler):
        # x_scaled = x * scale_ + min_  ->  x = x_scaled / scale_ - min_ / scale_
        return -scaler.min_ / scaler.scale_, 1.0 / scaler.scale_
    raise TypeError(type(scaler))


def scale_with(cls, X_tr, X_val, X_te, y_tr, y_val, y_te):
    sx, sy = cls().fit(X_tr), cls().fit(y_tr)
    return dict(X_tr=sx.transform(X_tr), X_val=sx.transform(X_val), X_te=sx.transform(X_te),
                y_tr=sy.transform(y_tr), y_val=sy.transform(y_val), y_te=sy.transform(y_te),
                scaler_X=sx, scaler_y=sy)


def train_explainable(d, feature_names, target_names, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_mean, x_std = affine_params(d['scaler_X'])
    y_mean, y_std = affine_params(d['scaler_y'])
    model = ExplainableDBPsModel(
        input_dim=d['X_tr'].shape[1], num_targets=d['y_tr'].shape[1],
        feature_names=feature_names, x_mean=x_mean, x_std=x_std)
    hier = HierarchicalConsistencyLoss(target_names, y_mean=y_mean, y_std=y_std,
                                       lambda_h=0.1, use_bcaa=True, nonneg=True)
    mse = torch.nn.MSELoss()
    train_torch_model(model, lambda p, t: mse(p, t) + hier(p),
                      d['X_tr'], d['y_tr'], d['X_val'], d['y_val'],
                      epochs=1000, patience=100)
    return _metrics_row(lambda X_: torch_predict(model, X_), d, target_names,
                        params=count_parameters(model))


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
            parts = split_train_val_test(X, y, seed)
            for sname, cls in SCALERS.items():
                t0 = time.time()
                d = scale_with(cls, *parts)
                r = train_explainable(d, feature_names, target_names, seed)
                rows.append({'Dataset': ds, 'Seed': seed, 'Scaler': sname,
                             'Model': 'Explainable Model', **r})
                m = fit_tuned('Support Vector Regression', d['X_tr'], d['y_tr'], seed)
                r2 = _metrics_row(m.predict, d, target_names)
                rows.append({'Dataset': ds, 'Seed': seed, 'Scaler': sname,
                             'Model': 'SVR', **r2})
                print(f"{ds} seed {seed} {sname:<7} explainable={r['Test R2']:.3f} "
                      f"svr={r2['Test R2']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'scaling_comparison_per_seed.csv'), index=False)

    out = []
    for (ds, model), g in df.groupby(['Dataset', 'Model']):
        piv = g.pivot(index='Seed', columns='Scaler', values='Test R2')
        for sname in SCALERS:
            gs = g[g['Scaler'] == sname]
            row = {'Dataset': ds, 'Model': model, 'Scaler': sname, 'N_seeds': len(gs),
                   'Test R2 Mean': gs['Test R2'].mean(),
                   'Test R2 Std': gs['Test R2'].std(ddof=0)}
            if sname != 'zscore':
                row['Delta vs zscore'] = (piv[sname] - piv['zscore']).mean()
                if len(piv) >= 3:
                    row['p_vs_zscore'] = stats.ttest_rel(piv[sname], piv['zscore']).pvalue
            out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(args.out, 'scaling_comparison_summary.csv'), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

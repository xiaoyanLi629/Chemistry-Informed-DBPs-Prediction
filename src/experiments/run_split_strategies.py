#!/usr/bin/env python3
"""Ontario split-strategy experiment .

Compares three evaluation protocols on the Ontario DWSP dataset:
1. random 70/30 splits (10 seeds) — the reference protocol;
2. GroupKFold by drinking-water system (5 folds) — no system appears in both
   train and test;
3. chronological split by sample date (earliest ~70% train, latest ~30% test),
   with 10 torch-init seeds.

Note: all 175 retained samples are from 2017 (the 1998-2024 range in the paper
refers to the surveillance program, not the filtered dataset), so cross-year
splits are impossible; the chronological split operates on sample dates within
2017 and the group split addresses the same-system leakage concern directly.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments.protocol import SEEDS_10
from experiments.run_benchmark import run_one_seed

FEATURES = ['Temp', 'pH', 'DOC', 'Cl2', 'NO2-N', 'NH4-N', 'Br']
TARGETS = ['DCAA', 'TCAA', 'HAA5', 'TTHM']
MODELS = ['Explainable Model', 'Support Vector Regression', 'Random Forest',
          'XGBoost', 'TabNet', 'Multi-task MLP']


def load():
    df = pd.read_csv('data/ontario_data_grouped.csv')
    X = df[FEATURES].values.astype(float)
    y = df[TARGETS].values.astype(float)
    groups = df['DWS_NAME'].values
    dates = df['SAMPLE_DATE'].values  # YYYYMMDD ints
    return X, y, groups, dates


def rows_from(res, protocol, fold):
    return [{'Protocol': protocol, 'Fold': fold, 'Model': m, **r} for m, r in res.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='1 seed / 1 fold smoke test')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, groups, dates = load()
    seeds = SEEDS_10[:1] if args.quick else SEEDS_10
    rows = []

    # 1. random splits (reference)
    for seed in seeds:
        t0 = time.time()
        res = run_one_seed(X, y, FEATURES, TARGETS, seed, models=MODELS)
        rows += rows_from(res, 'random', seed)
        print(f"random seed {seed}: {time.time()-t0:.0f}s", flush=True)

    # 2. GroupKFold by water system
    gkf = GroupKFold(n_splits=5)
    for fold, (trval_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        if args.quick and fold > 0:
            break
        # carve validation group-aware from the non-test systems
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=fold)
        tr_rel, val_rel = next(gss.split(X[trval_idx], y[trval_idx], groups[trval_idx]))
        tr_idx, val_idx = trval_idx[tr_rel], trval_idx[val_rel]
        t0 = time.time()
        res = run_one_seed(X, y, FEATURES, TARGETS, seed=42 + fold, models=MODELS,
                           splits=(tr_idx, val_idx, te_idx))
        rows += rows_from(res, 'group_kfold', fold)
        print(f"group fold {fold}: test systems={len(set(groups[te_idx]))}, "
              f"n_test={len(te_idx)}, {time.time()-t0:.0f}s", flush=True)

    # 3. chronological split by sample date
    order = np.argsort(dates, kind='stable')
    n = len(order)
    n_te = int(round(n * 0.3))
    trval_idx = order[:n - n_te]
    te_idx = order[n - n_te:]
    n_val = int(round(len(trval_idx) * 0.2))
    tr_idx, val_idx = trval_idx[:len(trval_idx) - n_val], trval_idx[len(trval_idx) - n_val:]
    cut_date = dates[te_idx].min()
    print(f"chronological: train<{cut_date}<=test, n_train={len(tr_idx)}, "
          f"n_val={len(val_idx)}, n_test={len(te_idx)}", flush=True)
    for seed in seeds:
        t0 = time.time()
        res = run_one_seed(X, y, FEATURES, TARGETS, seed, models=MODELS,
                           splits=(tr_idx, val_idx, te_idx))
        rows += rows_from(res, 'chronological', seed)
        print(f"chrono seed {seed}: {time.time()-t0:.0f}s", flush=True)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(args.out, 'ontario_split_strategies_per_fold.csv'),
                    index=False)

    summary = (per_fold.groupby(['Protocol', 'Model'])['Test R2']
               .agg(['mean', 'std', 'count']).reset_index()
               .sort_values(['Protocol', 'mean'], ascending=[True, False]))
    summary.to_csv(os.path.join(args.out, 'ontario_split_strategies_summary.csv'),
                   index=False)
    print(summary.to_string(index=False))

    # degradation table: random - group / random - chrono
    piv = summary.pivot(index='Model', columns='Protocol', values='mean')
    if {'random', 'group_kfold'} <= set(piv.columns):
        piv['delta_random_minus_group'] = piv['random'] - piv['group_kfold']
    if {'random', 'chronological'} <= set(piv.columns):
        piv['delta_random_minus_chrono'] = piv['random'] - piv['chronological']
    print(piv.round(3).to_string())


if __name__ == '__main__':
    main()

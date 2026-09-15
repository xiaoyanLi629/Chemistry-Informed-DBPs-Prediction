#!/usr/bin/env python3
"""Same-dataset learning-curve experiment .

Tests the "sample size is the primary bottleneck" hypothesis without
conflating it with cross-dataset differences: the training pool of a single
dataset is subsampled at increasing fractions with a fixed held-out test set,
so the effect of n is isolated from all other dataset differences.
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

from sklearn.model_selection import train_test_split

from experiments.protocol import SEEDS_10
from experiments.run_benchmark import load_dataset, run_one_seed

FRACTIONS = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
MODELS = ['Explainable Model', 'Support Vector Regression', 'Random Forest', 'XGBoost']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='both', choices=['original', 'ontario', 'both'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds_names = ['ontario', 'original'] if args.dataset == 'both' else [args.dataset]
    seeds = SEEDS_10[:1] if args.quick else SEEDS_10
    fractions = FRACTIONS[-2:] if args.quick else FRACTIONS

    rows = []
    for ds in ds_names:
        X, y, feature_names, target_names = load_dataset(ds)
        n = len(X)
        idx = np.arange(n)
        for seed in seeds:
            # fixed 30% test split for this seed; pool = the remaining 70%
            pool_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=seed)
            rng = np.random.default_rng(seed)
            for frac in fractions:
                n_sub = max(8, int(round(len(pool_idx) * frac)))
                sub = rng.choice(pool_idx, size=n_sub, replace=False)
                n_val = max(2, int(round(n_sub * 0.2)))
                val_idx, tr_idx = sub[:n_val], sub[n_val:]
                t0 = time.time()
                res = run_one_seed(X, y, feature_names, target_names, seed,
                                   models=MODELS, splits=(tr_idx, val_idx, te_idx))
                for m, r in res.items():
                    rows.append({'Dataset': ds, 'Seed': seed, 'Fraction': frac,
                                 'N_train': len(tr_idx) + len(val_idx), 'Model': m,
                                 'Test R2': r['Test R2']})
                print(f"{ds} seed {seed} frac {frac} (n={n_sub}): {time.time()-t0:.0f}s",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'learning_curve.csv'), index=False)

    summary = (df.groupby(['Dataset', 'Model', 'Fraction'])
               .agg(N_train=('N_train', 'mean'), R2_mean=('Test R2', 'mean'),
                    R2_std=('Test R2', 'std')).reset_index())
    summary.to_csv(os.path.join(args.out, 'learning_curve_summary.csv'), index=False)
    print(summary.round(3).to_string(index=False))
    plot(summary, args.out)


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    datasets = summary['Dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5),
                             squeeze=False)
    colors = {'Explainable Model': '#FF9800', 'Support Vector Regression': '#2196F3',
              'Random Forest': '#4CAF50', 'XGBoost': '#9C27B0'}
    for ax, ds in zip(axes[0], datasets):
        sub = summary[summary['Dataset'] == ds]
        for model, g in sub.groupby('Model'):
            g = g.sort_values('N_train')
            c = colors.get(model, 'gray')
            ax.plot(g['N_train'], g['R2_mean'], marker='o', label=model, color=c)
            ax.fill_between(g['N_train'], g['R2_mean'] - g['R2_std'],
                            g['R2_mean'] + g['R2_std'], alpha=0.15, color=c)
        ax.set_xlabel('Training samples (incl. validation)')
        ax.set_ylabel('Test R²')
        ax.set_title(f'{ds} dataset')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out, 'learning_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"saved {path}")


if __name__ == '__main__':
    main()

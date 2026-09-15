#!/usr/bin/env python3
"""Multi-seed module-level and feature-level ablation .

Two complementary ablations:
- module ablation: attention, chemistry features, interaction branch, chemical
  grouping (vs none / random), hierarchical loss, and a plain multi-task MLP
  control — each over 10 seeds with paired t-tests against the full model;
- feature ablation: leave-one-feature-out over 10 seeds;
- minimal sensor set: the paper's recommended DOC+UVA254+Temp subset trained
  directly (explainable model and tuned SVR).
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from experiments.baselines import fit_tuned
from experiments.protocol import SEEDS_10, scale_all, split_train_val_test
from experiments.run_benchmark import (load_dataset, train_explainable_model,
                                       train_mlp, _metrics_row)

MODULE_ARMS = {
    'full': {},
    'no_attention': {'use_attention': False},
    'no_chemistry': {'use_chemistry': False},
    'no_interaction': {'use_interaction': False},
    'no_grouping': {'grouping': 'none'},
    'random_grouping': {'grouping': 'random'},
    'no_hier_loss': {'lambda_h': 0.0},
}

MINIMAL_SENSORS = ['UVA254', 'DOC', 'Temp']


def normalize(name):
    from models import ChemistryInformedFeatureExtractor
    return ChemistryInformedFeatureExtractor._normalize_feature_name(name)


def run_module_ablation(ds, seeds, out_rows):
    X, y, feature_names, target_names = load_dataset(ds)
    for seed in seeds:
        parts = split_train_val_test(X, y, seed)
        d = scale_all(*parts)
        for arm, flags in MODULE_ARMS.items():
            t0 = time.time()
            kwargs = dict(flags)
            if arm == 'random_grouping':
                kwargs['rng_seed'] = seed
            row, _, _ = train_explainable_model(d, feature_names, target_names, seed,
                                                **kwargs)
            out_rows.append({'Dataset': ds, 'Experiment': 'module', 'Arm': arm,
                             'Seed': seed, **row})
            print(f"{ds} seed {seed} {arm}: R2={row['Test R2']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # parameter-comparable unstructured control
        row = train_mlp(d, target_names, seed)
        out_rows.append({'Dataset': ds, 'Experiment': 'module', 'Arm': 'multitask_mlp',
                         'Seed': seed, **row})


def run_feature_ablation(ds, seeds, out_rows):
    X, y, feature_names, target_names = load_dataset(ds)
    for seed in seeds:
        for i, fname in enumerate(['(none)'] + feature_names):
            if i == 0:
                X_abl, fnames = X, feature_names
                arm = 'baseline_all_features'
            else:
                X_abl = np.delete(X, i - 1, axis=1)
                fnames = [f for j, f in enumerate(feature_names) if j != i - 1]
                arm = f'remove_{normalize(fname)}'
            parts = split_train_val_test(X_abl, y, seed)
            d = scale_all(*parts)
            row, _, _ = train_explainable_model(d, fnames, target_names, seed)
            out_rows.append({'Dataset': ds, 'Experiment': 'feature', 'Arm': arm,
                             'Seed': seed, **row})
        print(f"{ds} feature ablation seed {seed} done", flush=True)


def run_minimal_sensors(seeds, out_rows):
    X, y, feature_names, target_names = load_dataset('original')
    norm = [normalize(f) for f in feature_names]
    keep = [i for i, f in enumerate(norm) if f in MINIMAL_SENSORS]
    assert len(keep) == 3, f"expected 3 minimal sensors, got {keep} from {norm}"
    X_min = X[:, keep]
    fnames_min = [feature_names[i] for i in keep]
    for seed in seeds:
        parts = split_train_val_test(X_min, y, seed)
        d = scale_all(*parts)
        row, _, _ = train_explainable_model(d, fnames_min, target_names, seed)
        out_rows.append({'Dataset': 'original', 'Experiment': 'minimal_sensors',
                         'Arm': 'explainable_minimal', 'Seed': seed, **row})
        m = fit_tuned('Support Vector Regression', d['X_tr'], d['y_tr'], seed)
        row = _metrics_row(m.predict, d, target_names)
        out_rows.append({'Dataset': 'original', 'Experiment': 'minimal_sensors',
                         'Arm': 'svr_minimal', 'Seed': seed, **row})
        print(f"minimal sensors seed {seed} done", flush=True)


def summarize(df, out_dir):
    frames = []
    for (ds, exp), g in df.groupby(['Dataset', 'Experiment']):
        base_arm = 'full' if exp == 'module' else 'baseline_all_features'
        base = g[g['Arm'] == base_arm].set_index('Seed')['Test R2'] \
            if base_arm in set(g['Arm']) else None
        for arm, ga in g.groupby('Arm'):
            row = {'Dataset': ds, 'Experiment': exp, 'Arm': arm,
                   'N_seeds': len(ga),
                   'Test R2 Mean': ga['Test R2'].mean(),
                   'Test R2 Std': ga['Test R2'].std(ddof=0)}
            if base is not None and arm != base_arm:
                paired = ga.set_index('Seed')['Test R2'].reindex(base.index).dropna()
                common = base.loc[paired.index]
                row['Delta R2 vs base'] = paired.mean() - base.mean()
                if len(paired) >= 3:
                    row['p_value_paired_t'] = stats.ttest_rel(paired, common).pvalue
            frames.append(row)
    summary = pd.DataFrame(frames).sort_values(['Dataset', 'Experiment', 'Test R2 Mean'],
                                               ascending=[True, True, False])
    summary.to_csv(os.path.join(out_dir, 'ablation_summary.csv'), index=False)
    print(summary.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--datasets', default='original,ontario')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = SEEDS_10[:1] if args.quick else SEEDS_10
    ds_list = args.datasets.split(',')
    rows = []
    for ds in ds_list:
        run_module_ablation(ds, seeds, rows)
        run_feature_ablation(ds, seeds, rows)
    run_minimal_sensors(seeds, rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'ablation_per_seed.csv'), index=False)
    summarize(df, args.out)


if __name__ == '__main__':
    main()

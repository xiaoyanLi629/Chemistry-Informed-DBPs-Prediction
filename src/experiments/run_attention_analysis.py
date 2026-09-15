#!/usr/bin/env python3
"""Attention-faithfulness analysis .

Attention weights are not automatically faithful explanations. This script
tests three properties on the original dataset:
1. stability — attention group weights across 10 independently trained seeds;
2. agreement with permutation importance — model-agnostic ΔR² per feature on
   the held-out test set, aggregated to the five chemical groups;
3. agreement with SHAP — KernelSHAP values aggregated to groups (3 seeds).

Outputs Spearman rank correlations between the three group-importance views
and the per-seed rank of the organic-matter group.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from sklearn.metrics import r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from experiments.protocol import (SEEDS_10, scale_all, split_train_val_test,
                                  torch_predict)
from experiments.run_benchmark import load_dataset, train_explainable_model


def group_of_features(model, n_features):
    """Map feature index -> group name using the trained model's grouping."""
    feat2group = {}
    for g, idxs in model.feature_extractor.feature_groups.items():
        for i in idxs:
            feat2group[i] = g
    return [feat2group.get(i) for i in range(n_features)]


def permutation_importance_grouped(model, d, groups, n_repeats=20, seed=0):
    rng = np.random.default_rng(seed)
    X_te, y_te = d['X_te'], d['y_te']
    base = r2_score(y_te, torch_predict(model, X_te))
    imp = {}
    for i, g in enumerate(groups):
        if g is None:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, i] = rng.permutation(Xp[:, i])
            drops.append(base - r2_score(y_te, torch_predict(model, Xp)))
        imp[g] = imp.get(g, 0.0) + float(np.mean(drops))
    return imp


def shap_importance_grouped(model, d, groups, n_background=10, n_samples=100):
    import shap
    f = lambda X_: torch_predict(model, np.asarray(X_, dtype=float))
    background = shap.kmeans(d['X_tr'], n_background)
    explainer = shap.KernelExplainer(f, background)
    sv = explainer.shap_values(d['X_te'], nsamples=n_samples, silent=True)
    # shap returns either a list of (N, F) arrays per target, or one array of
    # shape (N, F, T) (newer versions) / (N, F) (single output)
    if isinstance(sv, list):
        mean_abs = np.abs(np.stack(sv, axis=0)).mean(axis=(0, 1))  # (F,)
    else:
        arr = np.asarray(sv)
        mean_abs = (np.abs(arr).mean(axis=(0, 2)) if arr.ndim == 3
                    else np.abs(arr).mean(axis=0))
    imp = {}
    for i, g in enumerate(groups):
        if g is None:
            continue
        imp[g] = imp.get(g, 0.0) + float(mean_abs[i])
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--skip-shap', action='store_true')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, feature_names, target_names = load_dataset('original')
    seeds = SEEDS_10[:2] if args.quick else SEEDS_10
    shap_seeds = seeds[:1] if args.quick else SEEDS_10[:3]

    attn_rows, perm_rows, shap_rows = [], [], []
    group_order = None
    for seed in seeds:
        t0 = time.time()
        parts = split_train_val_test(X, y, seed)
        d = scale_all(*parts)
        row, model, _ = train_explainable_model(d, feature_names, target_names, seed)
        groups = group_of_features(model, X.shape[1])
        group_order = model.feature_extractor.active_groups

        with torch.no_grad():
            _, attn, _ = model(torch.FloatTensor(d['X_tr']), return_attention=True)
        w = attn.mean(dim=0).numpy()
        attn_rows.append({'Seed': seed, 'Test R2': row['Test R2'],
                          **{g: float(wi) for g, wi in zip(group_order, w)}})

        perm = permutation_importance_grouped(model, d, groups, seed=seed)
        perm_rows.append({'Seed': seed, **perm})

        if not args.skip_shap and seed in shap_seeds:
            sh = shap_importance_grouped(model, d, groups)
            shap_rows.append({'Seed': seed, **sh})
        print(f"seed {seed}: attention={dict(zip(group_order, w.round(3)))} "
              f"({time.time()-t0:.0f}s)", flush=True)

    attn_df = pd.DataFrame(attn_rows)
    perm_df = pd.DataFrame(perm_rows)
    attn_df.to_csv(os.path.join(args.out, 'attention_stability.csv'), index=False)

    # comparison table: mean group importance under each method (normalized to sum 1)
    def norm_mean(df, cols):
        v = df[cols].mean().clip(lower=0)
        return v / v.sum() if v.sum() > 0 else v

    comp = pd.DataFrame({'attention': norm_mean(attn_df, group_order),
                         'permutation': norm_mean(perm_df, group_order)})
    if shap_rows:
        shap_df = pd.DataFrame(shap_rows)
        comp['shap'] = norm_mean(shap_df, group_order)
    comp['attention_std_across_seeds'] = attn_df[group_order].std()
    comp['attention_cv'] = comp['attention_std_across_seeds'] / attn_df[group_order].mean()
    comp.to_csv(os.path.join(args.out, 'importance_comparison.csv'))

    # Spearman agreement between methods
    pairs = [('attention', 'permutation')]
    if 'shap' in comp.columns:
        pairs += [('attention', 'shap'), ('permutation', 'shap')]
    agreement = {}
    for a, b in pairs:
        rho, p = sps.spearmanr(comp[a], comp[b])
        agreement[f'spearman_{a}_vs_{b}'] = rho
    # how often is organic_matter the top attention group?
    top_counts = attn_df[group_order].idxmax(axis=1).value_counts()
    agreement['organic_matter_top_fraction'] = (
        top_counts.get('organic_matter', 0) / len(attn_df))
    pd.Series(agreement).to_csv(os.path.join(args.out, 'importance_agreement.csv'))

    print(comp.round(4).to_string())
    print(pd.Series(agreement).round(3).to_string())
    plot(comp, group_order, args.out)


def plot(comp, group_order, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = [c for c in ['attention', 'permutation', 'shap'] if c in comp.columns]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.2),
                             sharey=True)
    labels = [g.replace('_', '\n') for g in group_order]
    for ax, m in zip(np.atleast_1d(axes), methods):
        vals = comp.loc[group_order, m]
        err = comp.loc[group_order, 'attention_std_across_seeds'] if m == 'attention' else None
        ax.bar(labels, vals, yerr=err, capsize=3, color='#1976D2')
        ax.set_title(f'{m} (normalized)')
        ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'attention_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"saved {out}/attention_analysis.png")


if __name__ == '__main__':
    main()

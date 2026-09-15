#!/usr/bin/env python3
"""Generate the summary figures from the results CSVs.

Outputs (PDF, into manuscript/figure/):
- model_comparison.pdf   : test R2 per model on both datasets (bar + std)
- learning_curve.pdf     : within-dataset learning curves
- attention_analysis.pdf : attention vs permutation vs SHAP group importance
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)

SHORT = {
    'Support Vector Regression': 'SVR', 'Support Vector Regression+chem': 'SVR+chem',
    'Random Forest+chem': 'RF+chem', 'Ridge Regression': 'Ridge',
    'Ridge Regression+chem': 'Ridge+chem', 'Lasso Regression': 'Lasso',
    'K-Nearest Neighbors': 'KNN', 'Gradient Boosting': 'Grad. Boost.',
    'Linear Regression': 'Linear Reg.', 'Decision Tree': 'Dec. Tree',
    'Explainable Model': 'Explainable (Ours)',
}
TITLES = {'original': 'Dataset 1 (n=66)', 'ontario': 'Ontario DWSP (n=175)'}
ORANGE, GREY = '#ff9900', '#a6a6a6'
TEXTWIDTH_IN = 4.80  # LNCS \textwidth (12.2 cm); figures are drawn at 1:1
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'pdf.fonttype': 42})


def model_comparison(out_dir, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.6))
    for ax, ds in zip(axes, ['original', 'ontario']):
        s = pd.read_csv(os.path.join(out_dir, f'benchmark_{ds}_summary.csv'))
        s = s[s['Model'] != 'KAN Model'].sort_values('Test R2 Mean', ascending=True)
        names = [SHORT.get(m, m) for m in s['Model']]
        colors = [ORANGE if m == 'Explainable Model' else GREY for m in s['Model']]
        y = np.arange(len(s))
        ax.barh(y, s['Test R2 Mean'], xerr=s['Test R2 Std'], color=colors,
                error_kw=dict(elinewidth=0.8, capsize=2), height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_title(TITLES[ds])
        ax.set_xlabel('Test R$^2$ (mean $\\pm$ std, 10 seeds)')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'model_comparison.pdf'))
    plt.close(fig)


def learning_curve(out_dir, fig_dir):
    s = pd.read_csv(os.path.join(out_dir, 'learning_curve_summary.csv'))
    order = ['Explainable Model', 'Random Forest', 'Support Vector Regression', 'XGBoost']
    colors = {'Explainable Model': ORANGE, 'Random Forest': '#2ca02c',
              'Support Vector Regression': '#1f77b4', 'XGBoost': '#9467bd'}
    labels = {'Explainable Model': 'Explainable (Ours)', 'Random Forest': 'Random Forest',
              'Support Vector Regression': 'SVR', 'XGBoost': 'XGBoost'}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 1.95))
    for ax, ds in zip(axes, ['original', 'ontario']):
        g = s[s['Dataset'] == ds]
        for m in order:
            gm = g[g['Model'] == m].sort_values('N_train')
            if gm.empty:
                continue
            lw = 1.8 if m == 'Explainable Model' else 1.0
            ax.plot(gm['N_train'], gm['R2_mean'], '-o', color=colors[m], lw=lw, ms=2.5,
                    label=labels[m])
            ax.fill_between(gm['N_train'], gm['R2_mean'] - gm['R2_std'],
                            gm['R2_mean'] + gm['R2_std'], color=colors[m], alpha=0.12)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_title(TITLES[ds])
        ax.set_xlabel('Training samples (incl. validation)')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('Test R$^2$')
    axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='lower right', ncol=2, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'learning_curve.pdf'))
    plt.close(fig)


def attention_analysis(out_dir, fig_dir):
    comp = pd.read_csv(os.path.join(out_dir, 'importance_comparison.csv'), index_col=0)
    groups = ['environmental', 'organic_matter', 'disinfectant', 'nitrogen_species', 'halides']
    groups = [g for g in groups if g in comp.index]
    labels = {'environmental': 'Environ.', 'organic_matter': 'Organic',
              'disinfectant': 'Disinf.', 'nitrogen_species': 'Nitrogen',
              'halides': 'Halides'}
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    methods = [('attention', 'Attention (10 seeds)'), ('permutation', 'Permutation importance'),
               ('shap', 'KernelSHAP')]
    methods = [(m, t) for m, t in methods if m in comp.columns]
    fig, axes = plt.subplots(1, len(methods), figsize=(TEXTWIDTH_IN, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(groups))
    for ax, (m, title) in zip(axes, methods):
        vals = comp.loc[groups, m].values
        err = comp.loc[groups, 'attention_std_across_seeds'].values if m == 'attention' else None
        ax.bar(x, vals, color=colors[:len(groups)], yerr=err,
               error_kw=dict(elinewidth=0.8, capsize=2))
        ax.set_xticks(x)
        ax.set_xticklabels([labels[g] for g in groups], fontsize=6.5, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_title(title)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('Normalized importance')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'attention_analysis.pdf'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--fig-dir', default='manuscript/figure')
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    model_comparison(args.out, args.fig_dir)
    learning_curve(args.out, args.fig_dir)
    attention_analysis(args.out, args.fig_dir)
    print('figures written to', args.fig_dir)


if __name__ == '__main__':
    sys.exit(main())

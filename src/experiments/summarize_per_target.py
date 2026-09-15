#!/usr/bin/env python3
"""Per-target test R2 summary from the benchmark (no retraining).

Aggregates the per-seed benchmark CSVs into a model x target table (mean and
std over seeds), reports each model's rank per target, and prints a LaTeX
fragment, so that model behaviour can be compared target by target rather than
through a single aggregated score.
"""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)

DISPLAY = {
    'Explainable Model': 'Explainable Model (Ours)',
    'Support Vector Regression': 'SVR',
    'Support Vector Regression+chem': 'SVR+chem',
    'Random Forest': 'Random Forest',
    'Random Forest+chem': 'Random Forest+chem',
    'Ridge Regression+chem': 'Ridge+chem',
    'Ridge Regression': 'Ridge',
    'Extra Trees': 'Extra Trees',
    'CatBoost': 'CatBoost',
    'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM',
    'Multi-task MLP': 'Multi-task MLP',
    'TabNet': 'TabNet',
    'K-Nearest Neighbors': 'KNN',
    'Gradient Boosting': 'Grad. Boosting',
}


def clean_target(col):
    """'Test R2 DCAA (ug/L)' -> 'DCAA'."""
    name = col.replace('Test R2 ', '')
    return name.split(' (')[0].split('(')[0].strip()


def summarize(ds, out_dir):
    df = pd.read_csv(os.path.join(out_dir, f'benchmark_{ds}_per_seed.csv'))
    df = df[df['Model'] != 'KAN Model']
    target_cols = [c for c in df.columns if c.startswith('Test R2 ') and c != 'Test R2']
    targets = [clean_target(c) for c in target_cols]

    g = df.groupby('Model')
    mean = g[target_cols + ['Test R2']].mean()
    std = g[target_cols + ['Test R2']].std(ddof=0)
    mean.columns = targets + ['Overall']
    std.columns = targets + ['Overall']
    mean = mean.sort_values('Overall', ascending=False)
    std = std.loc[mean.index]

    ranks = mean.rank(ascending=False, method='min').astype(int)
    out = pd.concat({'mean': mean, 'std': std, 'rank': ranks}, axis=1)
    out.to_csv(os.path.join(out_dir, f'per_target_{ds}_summary.csv'))

    print(f"\n=== {ds}: per-target test R2 (mean over seeds), models ranked by overall ===")
    print(mean.round(3).to_string())
    print(f"\n--- rank of each model per target ({ds}) ---")
    print(ranks.to_string())

    # target difficulty: best achievable R2 and spread across models
    print(f"\n--- target difficulty ({ds}): best model R2 / median model R2 ---")
    for t in targets:
        print(f"  {t:<6} best={mean[t].max():.3f} ({mean[t].idxmax()})  "
              f"median={mean[t].median():.3f}")
    return mean, std, ranks, targets


def latex_rows(mean, std, targets, models):
    lines = []
    for m in models:
        if m not in mean.index:
            continue
        cells = []
        for t in targets + ['Overall']:
            v, s = mean.loc[m, t], std.loc[m, t]
            bold = abs(v - mean[t].max()) < 1e-12
            cell = f"${v:.3f}{{\\scriptstyle\\pm{s:.3f}}}$"
            if bold:
                cell = f"$\\mathbf{{{v:.3f}}}{{\\scriptstyle\\pm{s:.3f}}}$"
            cells.append(cell)
        name = DISPLAY.get(m, m)
        if m == 'Explainable Model':
            name = f"\\textbf{{{name}}}"
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--top', type=int, default=6,
                    help='number of top overall models to include besides ours')
    args = ap.parse_args()

    latex = []
    for ds in ['original', 'ontario']:
        mean, std, ranks, targets = summarize(ds, args.out)
        top = [m for m in mean.index if m != 'Explainable Model'][:args.top]
        models = ['Explainable Model'] + top
        latex.append(f"% ---- {ds}: targets = {targets} ----")
        latex.extend(latex_rows(mean, std, targets, models))

    path = os.path.join(args.out, 'per_target_table_rows.tex')
    with open(path, 'w') as f:
        f.write("\n".join(latex) + "\n")
    print(f"\nLaTeX rows written to {path}")
    print("\n".join(latex))


if __name__ == '__main__':
    sys.exit(main())

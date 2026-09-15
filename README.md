# Chemistry-Informed Explainable Deep Learning for Disinfection By-Products Prediction

A comprehensive machine learning framework for predicting disinfection by-products (DBPs) in drinking water treatment, featuring a chemistry-informed explainable deep learning model with chemistry-aware attention and hierarchical consistency loss.

## Highlights

- **20 configurations** compared: tuned classical ML, ensemble methods (incl. XGBoost/LightGBM/CatBoost), deep learning, KAN, TabNet, chemistry-augmented controls, and the proposed explainable model
- **Chemistry-informed architecture**: input features partitioned into 5 chemically motivated groups with explicit HOCl speciation, effective Cl2, and Cl2/DOC ratio features
- **Attention mechanism**: chemistry-aware group-level attention weights, interpretable and validated against established DBPs formation chemistry
- **Hierarchical consistency loss**: enforces the HAA aggregate hierarchy (DCAA + TCAA ≤ HAA5, HAA5 + BCAA ≤ HAA9, non-negativity) during training
- **Two datasets**: original dataset (n=66) and Ontario DWSP (n=175)
- **Rigorous evaluation**: leakage-free protocol (validation-based early stopping, tuned baselines, single terminal test evaluation), mean±std over 10 random seeds, plus GroupKFold and chronological validation

## Project Structure

```
├── src/
│   ├── models.py                 # ExplainableDBPsModel, feature extractor, attention, hierarchical loss
│   ├── preprocess_ontario.py     # Ontario DWSP data preprocessing
│   ├── experiments/              # leakage-free experiment suite
│   │   ├── protocol.py           #   train/val/test protocol, validation-based early stopping
│   │   ├── baselines.py          #   tuned baseline zoo + chemistry-feature augmentation
│   │   ├── nets.py               #   multi-task MLP control
│   │   ├── run_benchmark.py      #   10-seed benchmark, 20 configurations
│   │   ├── run_split_strategies.py   # GroupKFold / chronological validation
│   │   ├── run_learning_curve.py     # within-dataset learning curves
│   │   ├── run_ablation.py           # module- and feature-level ablations
│   │   ├── run_lambda_sensitivity.py # hierarchical-loss sweep + violation rates
│   │   ├── run_attention_analysis.py # attention stability, permutation, SHAP
│   │   ├── run_regularization_sweep.py  # dropout / weight-decay / capacity sweep
│   │   ├── run_mlp_hier.py           # hierarchical loss applied to the MLP baseline
│   │   ├── run_scaling_comparison.py # z-score vs RobustScaler vs MinMax
│   │   ├── summarize_per_target.py   # per-target R² tables from benchmark CSVs
│   │   └── make_figures.py           # figures from results/
│   └── legacy/
│       └── run_multi_seed.py     # conventional-protocol runner (protocol-sensitivity comparison)
├── data/
│   ├── data.csv                  # Dataset 1 (66 samples, 8 features, 5 targets)
│   ├── ontario_data.csv          # Ontario DWSP (175 samples, 7 features, 4 targets)
│   └── ontario_data_grouped.csv  # same 175 samples + water-system and sample-date metadata
├── results/                      # leakage-free experiment outputs (CSVs + figures)
├── results_legacy/
│   └── conventional_protocol/    # outputs of src/legacy/run_multi_seed.py (original/, ontario/)
├── analysis/eda/                 # exploratory data analysis scripts and outputs (Dataset 1)
├── tests/                        # pytest suite for protocol, model variants, baselines
├── requirements.txt
└── .gitignore
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Leakage-Free Benchmark (both datasets, 10 seeds, 20 configurations)

```bash
python src/experiments/run_benchmark.py
```

### Run the Full Experiment Suite

```bash
python src/experiments/run_split_strategies.py    # GroupKFold / chronological validation
python src/experiments/run_learning_curve.py      # within-dataset learning curves
python src/experiments/run_ablation.py         # module- and feature-level ablations
python src/experiments/run_lambda_sensitivity.py  # hierarchical-loss sweep + violation rates
python src/experiments/run_attention_analysis.py  # attention stability, permutation, SHAP
python src/experiments/run_regularization_sweep.py  # dropout / weight-decay / capacity sweep
python src/experiments/run_mlp_hier.py              # hierarchical loss on the multi-task MLP
python src/experiments/run_scaling_comparison.py    # standardization sensitivity
python src/experiments/summarize_per_target.py      # per-target R² tables (no training)
```

All outputs are written to `results/`. Run tests with `python -m pytest tests/ -q`.

## Datasets

**Original dataset** (`data/data.csv`):
- 66 treated water samples, 8 features, 5 HAA targets (DCAA, TCAA, BCAA, HAA5, HAA9)
- Features: Temp (°C), pH, UVA254 (cm⁻¹), Cl2 (mg/L), NO2-N (µg/L), DOC (mg/L), NH4-N (µg/L), Br (µg/L)
- All HAA targets in µg/L (the species–aggregate hierarchy DCAA+TCAA+BCAA ≤ HAA5 ≤ HAA9 holds in the raw values)

**Ontario DWSP dataset** (`data/ontario_data.csv`):
- 175 treated-water samples from the Ontario Drinking Water Surveillance Program (all from 2017 after feature-completeness filtering; 96 water systems)
- 7 features (no UVA254), 4 targets: DCAA, TCAA, HAA5, TTHM

## Explainable Model Architecture

Input features are partitioned into 5 chemically motivated groups:

| Group | Features | Chemical Role |
|-------|----------|---------------|
| Organic Matter | DOC, UVA254 | Primary organic precursors |
| Nitrogen Compounds | NO2-N, NH4-N | Competitive chlorine consumers |
| Environmental | Temp, pH | Reaction kinetics (Arrhenius, HOCl speciation) |
| Halides | Br | Brominated DBPs pathway |
| Disinfectant | Cl2 | Primary oxidizing agent |

Three chemistry-derived features are computed: HOCl fraction (pH-dependent speciation), effective Cl2, and Cl2/DOC ratio. A chemistry-aware attention mechanism learns group importance weights; a hierarchical consistency loss enforces the HAA aggregate hierarchy on the original concentration scale: DCAA + TCAA ≤ HAA5 (MCAA, MBAA and DBAA are not measured, so the measured species form a lower bound), HAA5 + BCAA ≤ HAA9 (BCAA is an HAA9 species), and non-negativity of all outputs.

## Key Results (leak-free protocol, mean±std over 10 random seeds)

All experiments (`src/experiments/`, results in `results/`) use a leakage-free
protocol: a validation split carved from training data drives early stopping, LR
scheduling, and hyperparameter tuning (inner 3-fold CV); the test set is evaluated
exactly once. A companion protocol-sensitivity experiment quantifies how much
conventional test-informed evaluation practices inflate certain model families.

**Original dataset (n=66):**

| Model | Test R² | Test MAE |
|-------|---------|----------|
| SVR (best, tuned) | 0.402±0.080 | 0.532±0.083 |
| Explainable Model (ours, 6th/20) | 0.291±0.172 | 0.588±0.096 |

**Ontario DWSP (n=175):**

| Model | Test R² | Test MAE |
|-------|---------|----------|
| CatBoost (best, tuned) | 0.603±0.071 | 0.445±0.076 |
| Explainable Model (ours, 5th/20) | 0.573±0.093 | 0.461±0.089 |

The protocol-sensitivity experiment shows that baselines whose model selection
touches the test set gain up to 0.17 R² under conventional protocols (TabNet:
0.501 strict vs. 0.668 conventional on Ontario), while the Explainable Model
benefits from principled validation-based early stopping, is the least overfit
of the five best-performing models on Ontario (train R² 0.718 vs. 0.909–0.965),
and attains the **best point estimate under leave-systems-out GroupKFold
validation** (R²=0.564, statistically tied with SVR; degradation vs. random
split only −0.009). Per target (`results/per_target_*_summary.csv`) it ranks
3rd of 19 on each Ontario HAA target (DCAA, TCAA, HAA5) and 11th on TTHM.
A capacity/regularization sweep (`results/regularization_sweep_summary.csv`)
shows that raising dropout to 0.3–0.5 is the only lever that narrows the
train–test gap; reducing model capacity does not help and weight decay is inert.

Note: after feature-completeness filtering, all 175 Ontario samples are from 2017
(the 1998–2024 range refers to the DWSP program); `data/ontario_data_grouped.csv`
preserves water-system and sample-date metadata for group/chronological splits.

## Learned Attention Weights (Dataset 1, mean over 10 seeds)

| Feature Group | Weight | Interpretation |
|---------------|--------|----------------|
| Organic Matter | 57.0% (top in 9/10 seeds) | Dominant precursor for HAA formation |
| Nitrogen Compounds | 20.3% | Competitive Cl2 consumption |
| Environmental | 9.2% | Kinetics and speciation control |
| Disinfectant | 8.2% | Narrow operational range in practice |
| Halides | 5.2% | Brominated species pathway |

The group ranking agrees exactly with permutation importance and KernelSHAP
(Spearman ρ = 1.0); 10-seed feature ablation confirms DOC (ΔR² = −0.208, p=0.001)
and UVA254 (−0.180, p=0.014) as the most critical monitoring parameters.

## Requirements

- Python >= 3.9
- PyTorch >= 1.9.0
- scikit-learn >= 1.0.0
- pandas, numpy, matplotlib, seaborn, scipy
- xgboost, lightgbm, catboost, pytorch-tabnet, pykan, shap (for the experiment suite)

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{dbps2026,
  title={Chemistry-Informed Explainable Deep Learning for Disinfection By-Products Prediction},
  booktitle={Lecture Notes in Computer Science},
  year={2026}
}
```

## License

This project is for academic research purposes.

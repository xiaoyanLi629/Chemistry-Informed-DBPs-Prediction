import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.baselines import SKLEARN_SPECS, add_chemistry_features, fit_tuned

FEATS = ['Temp', 'pH ', 'UVA254(cm)', 'Cl2 (mg/L)', 'NO2 -N (mg/L)', 'DOC (mg/L)',
         'NH4-N (ug/L)', 'Br (ug/L)']


def _data(n=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(size=(n, 3))
    return X, y


def test_all_specs_fit_predict_untuned():
    X, y = _data()
    for name in SKLEARN_SPECS:
        m = fit_tuned(name, X, y, seed=42, tune=False)
        pred = m.predict(X)
        assert pred.shape == y.shape, name


def test_tuning_runs_for_svr():
    X, y = _data()
    m = fit_tuned('Support Vector Regression', X, y, seed=42, tune=True)
    assert m.predict(X).shape == y.shape


def test_new_baselines_present():
    for name in ['XGBoost', 'LightGBM', 'CatBoost']:
        assert name in SKLEARN_SPECS


def test_add_chemistry_features():
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(20, 8))) + 0.5
    X_aug, names = add_chemistry_features(X, FEATS)
    assert X_aug.shape == (20, 11)
    assert names[-3:] == ['f_HOCl', 'effective_Cl2', 'Cl2_DOC']
    assert np.isfinite(X_aug).all()
    # f_HOCl in (0, 1)
    assert (X_aug[:, 8] > 0).all() and (X_aug[:, 8] < 1).all()


def test_add_chemistry_features_noop_when_missing():
    X = np.random.default_rng(2).normal(size=(10, 2))
    X_aug, names = add_chemistry_features(X, ['Temp', 'Br'])
    assert X_aug.shape == (10, 2)
    assert names == ['Temp', 'Br']

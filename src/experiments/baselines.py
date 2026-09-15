"""Baseline model zoo with light hyperparameter tuning and chemistry-feature augmentation.

Tuning uses GridSearchCV with an inner 3-fold CV *on the training split only*;
the parameter grid is searched on the first target and the chosen setting is
then fit for all targets via MultiOutputRegressor.
"""

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from models import ChemistryInformedFeatureExtractor

# name -> (factory(seed) -> estimator, param_grid)
SKLEARN_SPECS = {
    'Linear Regression': (
        lambda seed: LinearRegression(), {}),
    'Ridge Regression': (
        lambda seed: Ridge(alpha=1.0),
        {'alpha': [0.01, 0.1, 1.0, 10.0]}),
    'Lasso Regression': (
        lambda seed: Lasso(alpha=0.1, max_iter=10000),
        {'alpha': [0.01, 0.1, 1.0, 10.0]}),
    'Elastic Net': (
        lambda seed: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}),
    'Random Forest': (
        lambda seed: RandomForestRegressor(n_estimators=100, random_state=seed),
        {'n_estimators': [100, 300], 'max_depth': [None, 5, 10]}),
    'Extra Trees': (
        lambda seed: ExtraTreesRegressor(n_estimators=100, random_state=seed),
        {'n_estimators': [100, 300], 'max_depth': [None, 5, 10]}),
    'Gradient Boosting': (
        lambda seed: GradientBoostingRegressor(n_estimators=100, random_state=seed),
        {'n_estimators': [100, 300], 'max_depth': [2, 3], 'learning_rate': [0.05, 0.1]}),
    'Support Vector Regression': (
        lambda seed: SVR(kernel='rbf', C=1.0),
        {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': ['scale', 0.1]}),
    'K-Nearest Neighbors': (
        lambda seed: KNeighborsRegressor(n_neighbors=5),
        {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}),
    'Decision Tree': (
        lambda seed: DecisionTreeRegressor(random_state=seed),
        {'max_depth': [None, 3, 5, 10], 'min_samples_leaf': [1, 3, 5]}),
    'XGBoost': (
        lambda seed: XGBRegressor(n_estimators=100, random_state=seed, verbosity=0),
        {'n_estimators': [100, 300], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]}),
    'LightGBM': (
        lambda seed: LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1,
                                   min_child_samples=5),
        {'n_estimators': [100, 300], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]}),
    'CatBoost': (
        lambda seed: CatBoostRegressor(n_estimators=100, random_state=seed, verbose=0,
                                       allow_writing_files=False),
        {'n_estimators': [100, 300], 'depth': [3, 6], 'learning_rate': [0.05, 0.1]}),
}


def fit_tuned(name, X_tr, y_tr, seed, tune=True):
    """Fit one baseline; when tune=True, grid-search params with inner 3-fold CV on train only."""
    factory, grid = SKLEARN_SPECS[name]
    if not tune or not grid:
        m = MultiOutputRegressor(factory(seed))
        m.fit(X_tr, y_tr)
        return m
    gs = GridSearchCV(factory(seed), grid, cv=3, scoring='r2', n_jobs=8)
    gs.fit(X_tr, y_tr[:, 0])
    best = factory(seed)
    best.set_params(**gs.best_params_)
    m = MultiOutputRegressor(best)
    m.fit(X_tr, y_tr)
    return m


def add_chemistry_features(X, feature_names):
    """Append the model's chemistry-derived features (raw scale) as plain input columns.

    Used to test whether simple baselines benefit from the same domain knowledge:
    f_HOCl (pH-dependent HOCl fraction), effective_Cl2, and the Cl2/DOC ratio.
    X must be in the original (unscaled) feature space.
    """
    norm = [ChemistryInformedFeatureExtractor._normalize_feature_name(n) for n in feature_names]
    idx = {n: i for i, n in enumerate(norm)}
    cols, names = [], []
    if 'pH' in idx and 'Cl2' in idx:
        f_hocl = 1.0 / (1.0 + 10 ** (X[:, idx['pH']] - 7.5))
        cols.append(f_hocl)
        cols.append(X[:, idx['Cl2']] * f_hocl)
        names += ['f_HOCl', 'effective_Cl2']
    if 'Cl2' in idx and 'DOC' in idx:
        cols.append(X[:, idx['Cl2']] / (X[:, idx['DOC']] + 1e-8))
        names.append('Cl2_DOC')
    if not cols:
        return X, list(feature_names)
    return np.column_stack([X] + cols), list(feature_names) + names

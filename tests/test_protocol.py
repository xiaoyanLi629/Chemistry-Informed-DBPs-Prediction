import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.protocol import (SEEDS_10, count_parameters, evaluate, scale_all,
                                  split_train_val_test, train_torch_model)


def _toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    W = rng.normal(size=(4, 2))
    y = X @ W + 0.01 * rng.normal(size=(n, 2))
    return X, y


def test_split_sizes_and_disjoint():
    X, y = _toy_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_train_val_test(X, y, seed=42)
    assert len(X_te) == 60  # 30% of 200
    assert len(X_val) == 28  # 20% of remaining 140
    assert len(X_tr) == 112
    # disjoint: every row appears exactly once across the three splits
    all_rows = np.vstack([X_tr, X_val, X_te])
    assert all_rows.shape[0] == X.shape[0]
    assert np.allclose(np.sort(all_rows, axis=0), np.sort(X, axis=0))


def test_scale_all_fits_on_train_only():
    X, y = _toy_data()
    parts = split_train_val_test(X, y, seed=1)
    d = scale_all(*parts)
    # scaler mean must equal train mean, not global mean
    assert np.allclose(d['scaler_X'].mean_, parts[0].mean(axis=0))
    assert np.allclose(d['X_tr'].mean(axis=0), 0.0, atol=1e-9)


def test_train_torch_model_converges_and_restores_best():
    X, y = _toy_data()
    parts = split_train_val_test(X, y, seed=2)
    d = scale_all(*parts)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    loss_fn = torch.nn.MSELoss()
    hist = train_torch_model(model, loss_fn, d['X_tr'], d['y_tr'], d['X_val'], d['y_val'],
                             epochs=2000, patience=200, lr=1e-2)
    assert 'best_epoch' in hist and 'best_val_loss' in hist
    # restored weights reproduce the recorded best val loss
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(torch.FloatTensor(d['X_val'])),
                           torch.FloatTensor(d['y_val'])).item()
    assert abs(val_loss - hist['best_val_loss']) < 1e-6
    assert hist['best_val_loss'] < 0.05  # linear problem should fit well


def test_evaluate_per_target_keys():
    X, y = _toy_data()
    predict_fn = lambda X_: np.zeros((len(X_), 2))
    out = evaluate(predict_fn, X, y, target_names=['A', 'B'])
    for key in ['R2', 'MSE', 'MAE', 'R2_A', 'R2_B']:
        assert key in out


def test_count_parameters():
    model = torch.nn.Linear(4, 2)
    assert count_parameters(model) == 4 * 2 + 2


def test_seeds_10():
    assert len(SEEDS_10) == 10 and len(set(SEEDS_10)) == 10

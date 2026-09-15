"""Leak-free experimental protocol.

Train/validation/test discipline for all experiments: the validation set is
carved from the training split only; early stopping and LR scheduling see
validation loss only; the test set is touched exactly once, after training.
"""

import copy

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEEDS_10 = [42, 123, 456, 789, 1024, 2025, 3407, 5555, 7777, 9999]

# Tiny full-batch models train fastest single-threaded; avoids thread thrashing
# on many-core machines.
torch.set_num_threads(1)


def split_train_val_test(X, y, seed, test_size=0.3, val_size=0.2):
    """70/30 outer split, then 20% of the training side becomes validation."""
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=val_size, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def scale_all(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Standardize features and targets; scalers are fit on the training split only."""
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr)
    return dict(
        X_tr=sx.transform(X_tr), X_val=sx.transform(X_val), X_te=sx.transform(X_te),
        y_tr=sy.transform(y_tr), y_val=sy.transform(y_val), y_te=sy.transform(y_te),
        scaler_X=sx, scaler_y=sy)


def train_torch_model(model, loss_fn, X_tr, y_tr, X_val, y_val,
                      epochs=1000, patience=100, lr=1e-3, weight_decay=1e-5):
    """Full-batch Adam training with validation-based early stopping.

    LR scheduling (ReduceLROnPlateau) and early stopping both monitor the
    validation loss; the best-validation weights are restored before return.
    """
    device = torch.device('cpu')
    model.to(device)
    xt, yt = torch.FloatTensor(X_tr).to(device), torch.FloatTensor(y_tr).to(device)
    xv, yv = torch.FloatTensor(X_val).to(device), torch.FloatTensor(y_val).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=25, factor=0.5)
    best_val, best_state, best_epoch, bad = float('inf'), None, 0, 0
    history = {'train_loss': [], 'val_loss': []}
    epoch = -1
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        sched.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_state, best_epoch, bad = (
                val_loss, copy.deepcopy(model.state_dict()), epoch, 0)
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_val_loss=best_val, stopped_epoch=epoch)
    return history


def torch_predict(model, X):
    """Numpy-in / numpy-out prediction helper for a trained torch model."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).cpu().numpy()


def evaluate(predict_fn, X, y, target_names):
    """Overall and per-target regression metrics."""
    pred = predict_fn(X)
    out = {
        'R2': r2_score(y, pred),
        'MSE': mean_squared_error(y, pred),
        'MAE': mean_absolute_error(y, pred),
    }
    for j, t in enumerate(target_names):
        out[f'R2_{t}'] = r2_score(y[:, j], pred[:, j])
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

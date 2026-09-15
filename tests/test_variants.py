import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ExplainableDBPsModel, HierarchicalConsistencyLoss

FEATS = ['Temp', 'pH', 'UVA254', 'Cl2', 'NO2-N', 'DOC', 'NH4-N', 'Br']
TGTS = ['DCAA', 'TCAA', 'BCAA', 'HAA5', 'HAA9']


def _forward(**kw):
    m = ExplainableDBPsModel(input_dim=8, num_targets=5, feature_names=FEATS,
                             x_mean=np.zeros(8), x_std=np.ones(8), **kw)
    out = m(torch.randn(4, 8))
    assert out.shape == (4, 5)
    return m


def test_variants_forward():
    _forward()
    _forward(use_attention=False)
    _forward(use_chemistry=False)
    _forward(use_interaction=False)
    _forward(grouping='none')
    _forward(grouping='random', rng_seed=0)


def test_uniform_attention_when_disabled():
    m = ExplainableDBPsModel(input_dim=8, num_targets=5, feature_names=FEATS,
                             x_mean=np.zeros(8), x_std=np.ones(8), use_attention=False)
    _, attn, _ = m(torch.randn(4, 8), return_attention=True)
    n_groups = len(m.feature_extractor.active_groups)
    assert torch.allclose(attn, torch.full_like(attn, 1.0 / n_groups))


def test_random_grouping_differs_from_chemical():
    m1 = _forward(grouping='random', rng_seed=0)
    m2 = _forward()
    assert m1.feature_extractor.feature_groups != m2.feature_extractor.feature_groups
    # same group sizes though
    sizes = lambda m: sorted(len(v) for v in m.feature_extractor.feature_groups.values())
    assert sizes(m1) == sizes(m2)


def test_no_chemistry_removes_chem_branch():
    m = _forward(use_chemistry=False)
    assert m.feature_extractor.chem_out_dim == 0
    out = m(torch.randn(3, 8), return_attention=True)
    assert 'chemistry' not in out[2]


def test_hierarchical_loss_bcaa_and_nonneg():
    ln = HierarchicalConsistencyLoss(TGTS, y_mean=np.zeros(5), y_std=np.ones(5),
                                     lambda_h=1.0, use_bcaa=True, nonneg=True)
    ok = torch.tensor([[1.0, 1.0, 1.0, 4.0, 5.0]])    # 1+1<=4 and 4+1<=5
    bad = torch.tensor([[3.0, 2.0, 1.0, 4.0, 3.0]])   # 3+2=5 > HAA5=4 and 4+1=5 > HAA9=3
    # BCAA is an HAA9 species: a large BCAA must not trigger the HAA5 constraint
    bcaa_only = torch.tensor([[1.0, 1.0, 9.0, 4.0, 14.0]])
    assert ln(bcaa_only).item() == 0.0
    assert ln.violation_stats(bcaa_only)['viol_haa5'] == 0.0
    neg = torch.tensor([[-1.0, 1.0, 1.0, 4.0, 5.0]])
    assert ln(ok).item() == 0.0
    assert ln(bad).item() > 0.0
    assert ln(neg).item() > 0.0
    stats = ln.violation_stats(bad)
    assert stats['viol_haa5'] == 1.0 and stats['viol_haa9'] == 1.0
    assert ln.violation_stats(ok)['viol_haa5'] == 0.0


def test_hierarchical_loss_without_bcaa_matches_old_behavior():
    ln = HierarchicalConsistencyLoss(TGTS, y_mean=np.zeros(5), y_std=np.ones(5),
                                     lambda_h=1.0, use_bcaa=False, nonneg=False)
    # 2+2=4 <= 4 satisfied without BCAA (old constraint), 5 <= HAA9 ok
    pred = torch.tensor([[2.0, 2.0, 9.0, 4.0, 5.0]])
    assert ln(pred).item() == 0.0


def test_backward_compat_defaults():
    m = ExplainableDBPsModel(input_dim=8, num_targets=5, feature_names=FEATS,
                             x_mean=np.zeros(8), x_std=np.ones(8))
    assert m(torch.randn(2, 8)).shape == (2, 5)
    ln = HierarchicalConsistencyLoss(TGTS, y_mean=np.zeros(5), y_std=np.ones(5))
    assert ln(torch.randn(2, 5)).item() >= 0.0

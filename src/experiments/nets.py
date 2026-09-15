"""Neural baselines for the experiments."""

import torch


class MultiTaskMLP(torch.nn.Module):
    """Plain shared-trunk multi-task MLP baseline.

    Isolates the benefit of the group-structured attention architecture: same
    multi-task output design, but no chemical grouping, no chemistry features,
    no attention.
    """

    def __init__(self, input_dim, num_targets, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, 1) for _ in range(num_targets)])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)

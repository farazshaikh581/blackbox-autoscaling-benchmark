"""Per-minute autoscaling policy: a small MLP as the decision variable.

The dynamic variant replaces the static per-tier replica count with a
per-minute policy: a function of observed state (previous minute's load,
current replica count, and a reserved input that is always zero) to this
minute's target replica level. The policy's WEIGHTS are the decision
variable NSGA-II/MOEA-D search directly (neuroevolution): a flat real
vector, decoded here into one small feed-forward net per tier.

State is expected pre-normalized by the caller (`simulator.simulate_policy`,
which has the tier's own scale, capacity and replica bounds, to normalize
against); this module only knows about weight-vector
mechanics, not domain-specific scaling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

STATE_DIM = 3    # [lam_prev_norm, replicas_prev_norm, reserved (0)], each in [0, 1]
HIDDEN_DIM = 6


def weight_vector_size(state_dim: int = STATE_DIM, hidden_dim: int = HIDDEN_DIM) -> int:
    """Parameter count for one tier's net (W1, b1, W2, b2)."""
    return hidden_dim * state_dim + hidden_dim + hidden_dim + 1


@dataclass
class PolicyNet:
    """One tier's per-minute policy: normalized state -> target replica fraction.

    A single hidden layer (tanh), sigmoid output. The sigmoid output means raw
    EA-searched weights can never produce a runaway or negative target -- the
    output is always a bounded fraction of [0, 1], which the caller maps onto
    the tier's actual [replica_min, replica_max] range.

    `act` is called once per tier per simulated minute (a day is 1440 calls),
    so it's a real hot path -- weights are plain Python floats/lists rather
    than numpy arrays. numpy's per-call dispatch overhead dominates at this
    matrix size (hidden_dim=6, state_dim=3); plain `math.tanh`/`math.exp`
    over nested lists profiled several times faster here than `w1 @ state`.
    """

    w1: List[List[float]]   # (hidden_dim, state_dim)
    b1: List[float]         # (hidden_dim,)
    w2: List[float]         # (hidden_dim,)
    b2: float

    def act(self, state: Sequence[float]) -> float:
        z = self.b2
        for row, b1i, w2i in zip(self.w1, self.b1, self.w2):
            acc = b1i
            for wij, si in zip(row, state):
                acc += wij * si
            z += w2i * math.tanh(acc)
        return 1.0 / (1.0 + math.exp(-z))


def decode(
    flat_weights: np.ndarray,
    n_tiers: int,
    state_dim: int = STATE_DIM,
    hidden_dim: int = HIDDEN_DIM,
) -> List[PolicyNet]:
    """Slice a flat real vector (the EA's decision variable) into n_tiers nets.

    Uses numpy for the one-time slicing/reshaping (this runs once per
    evaluation, not once per minute), then converts to plain Python for
    `PolicyNet`'s hot-path `act`.
    """
    per_tier = weight_vector_size(state_dim, hidden_dim)
    flat_weights = np.asarray(flat_weights, dtype=float)
    if flat_weights.shape[0] != per_tier * n_tiers:
        raise ValueError(
            f"expected {per_tier * n_tiers} weights for {n_tiers} tiers "
            f"(state_dim={state_dim}, hidden_dim={hidden_dim}), got "
            f"{flat_weights.shape[0]}"
        )
    nets = []
    for i in range(n_tiers):
        chunk = flat_weights[i * per_tier:(i + 1) * per_tier]
        off = hidden_dim * state_dim
        w1 = chunk[:off].reshape(hidden_dim, state_dim)
        b1 = chunk[off:off + hidden_dim]
        off += hidden_dim
        w2 = chunk[off:off + hidden_dim]
        off += hidden_dim
        b2 = float(chunk[off])
        nets.append(PolicyNet(w1=w1.tolist(), b1=b1.tolist(), w2=w2.tolist(), b2=b2))
    return nets

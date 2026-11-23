"""Temporal smoothing with a three-state hidden Markov model.

The classifier gives p(state | frame) independently for every frame. We turn
those into scaled likelihoods p(state | frame) / p(state), raise them to a
power `alpha` (how much to trust the classifier) and decode the most likely
state sequence with Viterbi. The transition matrix is estimated from the
training labels and made "stickier" by dividing the switching probabilities
by `stickiness`. Both knobs are tuned on out-of-fold predictions from the
training match only.
"""
from __future__ import annotations

import numpy as np

N_STATES = 3


def transition_matrix(labels: np.ndarray, breaks: np.ndarray | None = None) -> np.ndarray:
    labels = np.asarray(labels)
    counts = np.ones((N_STATES, N_STATES))
    keep = np.ones(len(labels) - 1, dtype=bool)
    if breaks is not None:
        keep &= ~np.asarray(breaks, dtype=bool)[1:]
    np.add.at(counts, (labels[:-1][keep], labels[1:][keep]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def make_sticky(trans: np.ndarray, stickiness: float) -> np.ndarray:
    off = trans * (1 - np.eye(N_STATES)) / stickiness
    return off + np.diag(1 - off.sum(axis=1))


def viterbi(log_emission: np.ndarray, log_trans: np.ndarray, log_start: np.ndarray) -> np.ndarray:
    n = len(log_emission)
    score = log_start + log_emission[0]
    back = np.empty((n, N_STATES), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] + log_trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(proba: np.ndarray, prior: np.ndarray, trans: np.ndarray, alpha: float,
           segments: np.ndarray) -> np.ndarray:
    """Viterbi over each segment (e.g. each period) separately."""
    log_em = alpha * (np.log(np.clip(proba, 1e-6, 1)) - np.log(prior))
    log_trans = np.log(trans)
    log_start = np.log(prior)
    out = np.empty(len(proba), dtype=np.int8)
    for seg in np.unique(segments):
        sel = np.flatnonzero(segments == seg)
        out[sel] = viterbi(log_em[sel], log_trans, log_start)
    return out

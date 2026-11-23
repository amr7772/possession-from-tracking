"""The obvious baseline: whoever has the player closest to the ball has it.

The ball counts as dead when it is not tracked or sits outside the lines.
"""
from __future__ import annotations

import warnings

import numpy as np

from .data import PITCH_LENGTH, PITCH_WIDTH, Match
from .labels import AWAY, DEAD, HOME


def nearest_player(m: Match) -> np.ndarray:
    bx, by = m.ball[:, 0], m.ball[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dh = np.nanmin(np.hypot(m.home[..., 0] - bx[:, None], m.home[..., 1] - by[:, None]), axis=1)
        da = np.nanmin(np.hypot(m.away[..., 0] - bx[:, None], m.away[..., 1] - by[:, None]), axis=1)
    pred = np.where(dh <= da, HOME, AWAY).astype(np.int8)
    dead = np.isnan(bx) | (np.abs(bx) > PITCH_LENGTH / 2) | (np.abs(by) > PITCH_WIDTH / 2)
    pred[dead] = DEAD
    return pred

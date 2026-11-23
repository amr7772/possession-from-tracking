"""Frame classifier (LightGBM) and the smoothing step on top of it."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .labels import CLASS_NAMES
from .metrics import evaluate_segments
from .smoothing import decode, make_sticky, transition_matrix

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
}
N_ROUNDS = 400
STICKINESS_GRID = (1, 2, 4, 8, 16, 32, 64)
ALPHA_GRID = (0.4, 0.6, 0.8, 1.0)


def train_classifier(X: pd.DataFrame, y: np.ndarray, rounds: int = N_ROUNDS) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(X, label=y), num_boost_round=rounds)


def out_of_fold_proba(X: pd.DataFrame, y: np.ndarray, n_folds: int = 5, buffer: int = 50) -> np.ndarray:
    """Predictions for every row from a model that never saw its time block.

    Rows are split into contiguous blocks; `buffer` rows on each side of the
    held-out block are dropped from training so rolling windows do not leak.
    """
    n = len(X)
    proba = np.zeros((n, 3))
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        train = np.ones(n, dtype=bool)
        train[max(lo - buffer, 0): min(hi + buffer, n)] = False
        booster = train_classifier(X[train], y[train])
        proba[lo:hi] = booster.predict(X.iloc[lo:hi])
    return proba


@dataclass
class Smoother:
    prior: list
    trans: list
    alpha: float
    stickiness: float
    tuning: list = field(default_factory=list)

    def decode(self, proba: np.ndarray, segments: np.ndarray) -> np.ndarray:
        return decode(proba, np.asarray(self.prior), np.asarray(self.trans), self.alpha, segments)


def tune_smoother(proba, y, mask, period) -> Smoother:
    """Pick alpha and stickiness on out-of-fold predictions of the training match."""
    keep = np.asarray(mask, dtype=bool)
    prior = np.bincount(y[keep], minlength=3) / keep.sum()
    breaks = np.r_[True, period[1:] != period[:-1]]
    base_trans = transition_matrix(y, breaks)
    best, rows = None, []
    for stick, alpha in itertools.product(STICKINESS_GRID, ALPHA_GRID):
        trans = make_sticky(base_trans, stick)
        pred = decode(proba, prior, trans, alpha, period)
        s = evaluate_segments(y, pred, mask, period)
        objective = (s["macro_f1"] + s["spell_f1"]) / 2
        rows.append({"stickiness": stick, "alpha": alpha, "macro_f1": round(s["macro_f1"], 4),
                     "spell_f1": round(s["spell_f1"], 4), "objective": round(objective, 4)})
        if best is None or objective > best[0]:
            best = (objective, stick, alpha, trans)
    _, stick, alpha, trans = best
    return Smoother(prior=prior.tolist(), trans=trans.tolist(), alpha=alpha,
                    stickiness=stick, tuning=rows)


def save(booster: lgb.Booster, smoother: Smoother, features: list[str], step: int,
         out_dir: Path, extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "possession_lgbm.txt"))
    config = {
        "classes": list(CLASS_NAMES),
        "features": features,
        "sample_every_n_frames": step,
        "fps": 25 // step,
        "smoother": {"prior": smoother.prior, "transition": smoother.trans,
                     "alpha": smoother.alpha, "stickiness": smoother.stickiness},
    }
    if extra:
        config.update(extra)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def load(model_dir: Path) -> tuple[lgb.Booster, Smoother, dict]:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    booster = lgb.Booster(model_file=str(Path(model_dir) / "possession_lgbm.txt"))
    s = config["smoother"]
    return booster, Smoother(s["prior"], s["transition"], s["alpha"], s["stickiness"]), config

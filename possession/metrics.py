"""Scoring.

Two numbers matter:

* frame macro-F1: F1 for home, away and dead ball, averaged. It rewards being
  right most of the time but does not care whether a prediction flickers.
* spell F1: a possession spell is an unbroken run of one team holding the ball.
  A predicted spell matches a true spell of the same team when their
  intersection-over-union in time is at least 0.5, one-to-one. Precision is
  matched / predicted, recall is matched / true. A model that flickers between
  teams breaks spells into pieces and gets punished here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .labels import AWAY, DEAD, HOME

LABELS = (HOME, AWAY, DEAD)


def spells(labels: np.ndarray, breaks: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Runs of HOME or AWAY as (start, end_exclusive, team). `breaks` marks
    positions where a new run must start (e.g. a new period)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    change = np.r_[True, labels[1:] != labels[:-1]]
    if breaks is not None:
        change |= np.asarray(breaks, dtype=bool)
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(labels)]
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends) if labels[s] != DEAD]


def match_spells(true_sp, pred_sp, iou: float = 0.5) -> int:
    """Greedy one-to-one matching of spells by temporal IoU."""
    pairs = []
    j0 = 0
    for i, (ts, te, tt) in enumerate(true_sp):
        while j0 < len(pred_sp) and pred_sp[j0][1] <= ts:
            j0 += 1
        j = j0
        while j < len(pred_sp) and pred_sp[j][0] < te:
            ps, pe, pt = pred_sp[j]
            if pt == tt:
                inter = min(te, pe) - max(ts, ps)
                union = max(te, pe) - min(ts, ps)
                score = inter / union
                if score >= iou:
                    pairs.append((score, i, j))
            j += 1
    pairs.sort(reverse=True)
    used_t, used_p = set(), set()
    for _, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
    return len(used_t)


def spell_scores(y_true, y_pred, breaks=None, iou: float = 0.5) -> dict:
    ts, ps = spells(y_true, breaks), spells(y_pred, breaks)
    matched = match_spells(ts, ps, iou)
    precision = matched / len(ps) if ps else 0.0
    recall = matched / len(ts) if ts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"spell_f1": f1, "spell_precision": precision, "spell_recall": recall,
            "spells_true": len(ts), "spells_pred": len(ps), "spells_matched": matched}


def frame_scores(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_class)),
        "f1_home": float(per_class[0]),
        "f1_away": float(per_class[1]),
        "f1_dead": float(per_class[2]),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }


def score(y_true, y_pred, breaks=None) -> dict:
    out = frame_scores(y_true, y_pred)
    out.update(spell_scores(y_true, y_pred, breaks))
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()
    return out


def evaluate_segments(y_true, y_pred, mask, period) -> dict:
    """Score only frames inside `mask`; spells never cross a gap in the mask or a period change."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    yt, yp, per = np.asarray(y_true)[idx], np.asarray(y_pred)[idx], np.asarray(period)[idx]
    breaks = np.r_[True, (np.diff(idx) > 1) | (per[1:] != per[:-1])]
    return score(yt, yp, breaks)

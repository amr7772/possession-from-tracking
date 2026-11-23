import numpy as np

from possession.labels import AWAY, DEAD, HOME
from possession.metrics import frame_scores, spell_scores, spells

H, A, D = HOME, AWAY, DEAD


def test_spells_skip_dead_and_split_on_breaks():
    y = np.array([H, H, D, A, A, A, H])
    assert spells(y) == [(0, 2, H), (3, 6, A), (6, 7, H)]
    breaks = np.zeros(len(y), dtype=bool)
    breaks[4] = True
    assert spells(y, breaks) == [(0, 2, H), (3, 4, A), (4, 6, A), (6, 7, H)]


def test_perfect_prediction():
    y = np.array([H] * 10 + [A] * 10 + [D] * 5 + [H] * 10)
    s = spell_scores(y, y)
    assert s["spell_f1"] == 1.0
    assert s["spells_true"] == s["spells_matched"] == 3


def test_flicker_breaks_spells():
    y = np.array([H] * 20)
    s = spell_scores(y, np.array([H, A] * 10))
    assert s["spells_matched"] == 0
    assert s["spells_pred"] == 20


def test_iou_threshold():
    y = np.array([H] * 10 + [A] * 10)
    late = np.array([A] * 4 + [H] * 6 + [A] * 10)        # home spell IoU 0.6
    assert spell_scores(y, late)["spells_matched"] == 2
    too_late = np.array([A] * 6 + [H] * 4 + [A] * 10)    # home spell IoU 0.4
    assert spell_scores(y, too_late)["spells_matched"] == 1


def test_frame_macro_f1():
    f = frame_scores(np.array([H, A, D, D]), np.array([H, A, D, H]))
    assert abs(f["macro_f1"] - (2 / 3 + 1 + 2 / 3) / 3) < 1e-9

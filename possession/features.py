"""Per-frame features built from tracking data only. No events are used here.

Everything is computed at the native 25 Hz and the model later samples every
fifth frame. Windows look up to a few seconds both backwards and forwards: this
is an offline labeller, it is allowed to see what happens next.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

from .data import FPS, PITCH_LENGTH, PITCH_WIDTH, Match
from .labels import AWAY, HOME, _runs

HALF_L, HALF_W = PITCH_LENGTH / 2, PITCH_WIDTH / 2
ON_BALL_M = 1.5      # a player this close to the ball is "on it"
TOUCH_RADIUS_M = 3.0  # a touch needs a player within this distance
TOUCH_DV = 3.0       # m/s change in ball velocity across 0.16 s
CAP_S = 30.0         # cap for "seconds since / until" features


# ---------------------------------------------------------------- helpers

def interpolate_short_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN gaps of at most `max_gap` samples, along axis 0."""
    out = np.array(a, dtype=float, copy=True)
    flat = out.reshape(len(out), -1)
    idx = np.arange(len(out))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        nan = np.isnan(col)
        if not nan.any() or nan.all():
            continue
        filled = np.interp(idx, idx[~nan], col[~nan])
        for s, e in zip(*_runs(nan)):
            if e - s <= max_gap and s > 0 and e < len(col):
                col[s:e] = filled[s:e]
    return out


def smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that keeps NaN where the input was NaN."""
    flat = a.reshape(len(a), -1)
    sm = pd.DataFrame(flat).rolling(window, center=True, min_periods=1).mean().to_numpy(copy=True)
    sm[np.isnan(flat)] = np.nan
    return sm.reshape(a.shape)


def roll(x: np.ndarray, seconds: float, how: str = "mean", side: str = "past") -> np.ndarray:
    """Rolling statistic over the last (or next) `seconds`, current frame included."""
    w = max(1, int(round(seconds * FPS)))
    s = pd.Series(np.asarray(x, dtype=float))
    if side == "future":
        s = s.iloc[::-1].reset_index(drop=True)
    r = getattr(s.rolling(w, min_periods=1), how)().to_numpy(copy=True)
    return r[::-1].copy() if side == "future" else r


def seconds_since(event: np.ndarray) -> np.ndarray:
    idx = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, idx, -1))
    out = np.where(last >= 0, (idx - last) / FPS, CAP_S)
    return np.minimum(out, CAP_S)


def seconds_until(event: np.ndarray) -> np.ndarray:
    return seconds_since(event[::-1])[::-1].copy()


def attack_direction(m: Match) -> np.ndarray:
    """+1 where the home team attacks towards +x, -1 otherwise (per period)."""
    d = np.ones(len(m))
    for p in np.unique(m.period):
        sel = m.period == p
        cx = np.nanmean(m.home[sel, :, 0])
        d[sel] = 1.0 if cx < 0 else -1.0
    return d


def _proximity(players: np.ndarray, vel: np.ndarray, ball: np.ndarray, ball_v: np.ndarray):
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    dd = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(dd, axis=1)
    rows = np.arange(len(dd))
    i1 = order[:, 0]
    d1 = dd[rows, i1]
    d2 = dd[rows, order[:, 1]]
    d1[np.isinf(d1)] = np.nan
    d2[np.isinf(d2)] = np.nan
    v1 = vel[rows, i1]
    missing = np.isnan(ball[:, 0])
    n2 = np.where(missing, np.nan, (dd < 2.0).sum(axis=1))
    n5 = np.where(missing, np.nan, (dd < 5.0).sum(axis=1))
    return {
        "d1": d1,
        "d2": d2,
        "n2": n2,
        "n5": n5,
        "rel": np.hypot(*(ball_v - v1).T),   # ball speed relative to the nearest player
        "s1": np.hypot(*v1.T),               # speed of the nearest player
    }


# ---------------------------------------------------------------- touches

def detect_touches(dv: np.ndarray, d1h: np.ndarray, d1a: np.ndarray) -> pd.DataFrame:
    """Frames where the ball changes velocity sharply next to a player.

    Returns one row per touch with the frame index and the team of the closest
    player. These are the "touch detected" moments in the replay.
    """
    dv0 = np.nan_to_num(dv, nan=0.0)
    h = np.nan_to_num(d1h, nan=np.inf)
    a = np.nan_to_num(d1a, nan=np.inf)
    peak = (dv0 == maximum_filter1d(dv0, size=9)) & (dv0 > TOUCH_DV)
    peak &= np.fmin(h, a) < TOUCH_RADIUS_M
    idx = np.flatnonzero(peak)
    team = np.where(h[idx] <= a[idx], HOME, AWAY)
    return pd.DataFrame({"idx": idx, "team": team})


def _touch_context(touches: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    idx = np.arange(n)
    t_idx = touches["idx"].to_numpy()
    sign = np.where(touches["team"].to_numpy() == HOME, 1.0, -1.0)
    out: dict[str, np.ndarray] = {}
    if len(t_idx) == 0:
        for k in ("last_touch", "prev_touch", "next_touch"):
            out[k] = np.zeros(n)
        out["t_since_touch"] = np.full(n, CAP_S)
        out["t_to_touch"] = np.full(n, CAP_S)
        for k in ("touch_h_past5", "touch_a_past5", "touch_h_next5", "touch_a_next5"):
            out[k] = np.zeros(n)
        return out
    last = np.searchsorted(t_idx, idx, side="right") - 1
    nxt = last + 1
    has_last, has_prev, has_next = last >= 0, last >= 1, nxt < len(t_idx)
    out["last_touch"] = np.where(has_last, sign[np.clip(last, 0, None)], 0.0)
    out["prev_touch"] = np.where(has_prev, sign[np.clip(last - 1, 0, None)], 0.0)
    out["next_touch"] = np.where(has_next, sign[np.clip(nxt, None, len(t_idx) - 1)], 0.0)
    out["t_since_touch"] = np.minimum(
        np.where(has_last, (idx - t_idx[np.clip(last, 0, None)]) / FPS, CAP_S), CAP_S)
    out["t_to_touch"] = np.minimum(
        np.where(has_next, (t_idx[np.clip(nxt, None, len(t_idx) - 1)] - idx) / FPS, CAP_S), CAP_S)
    for team, tag in ((HOME, "h"), (AWAY, "a")):
        hit = np.zeros(n)
        hit[t_idx[touches["team"].to_numpy() == team]] = 1.0
        out[f"touch_{tag}_past5"] = roll(hit, 5, "sum", "past")
        out[f"touch_{tag}_next5"] = roll(hit, 5, "sum", "future")
    return out


# ---------------------------------------------------------------- main

def compute_features(m: Match) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features at 25 Hz, detected touches)."""
    n = len(m)
    raw_missing = np.isnan(m.ball[:, 0])

    ball = interpolate_short_gaps(m.ball, max_gap=FPS // 2)  # bridge short occlusions
    ball_s = smooth(ball, 5)
    bv = np.gradient(ball_s, axis=0) * FPS
    bspeed = np.hypot(*bv.T)
    k = 2
    dv = np.full(n, np.nan)
    dv[k:-k] = np.hypot(*(bv[2 * k:] - bv[:-2 * k]).T)

    H, A = smooth(m.home, 5), smooth(m.away, 5)
    Hv, Av = np.gradient(H, axis=0) * FPS, np.gradient(A, axis=0) * FPS
    Hs, As = np.hypot(Hv[..., 0], Hv[..., 1]), np.hypot(Av[..., 0], Av[..., 1])

    ph = _proximity(H, Hv, ball_s, bv)
    pa = _proximity(A, Av, ball_s, bv)
    touches = detect_touches(dv, ph["d1"], pa["d1"])

    direction = attack_direction(m)
    bx, by = ball_s[:, 0], ball_s[:, 1]
    missing = np.isnan(bx)
    out_of_lines = (np.abs(bx) > HALF_L + 0.3) | (np.abs(by) > HALF_W + 0.3)
    still = np.nan_to_num(bspeed, nan=99.0) < 0.5

    f: dict[str, np.ndarray] = {}
    # ball state
    f["ball_missing"] = raw_missing.astype(float)
    f["missing_for"] = np.where(raw_missing, seconds_since(~raw_missing), 0.0)
    f["missing_until"] = np.where(raw_missing, seconds_until(~raw_missing), 0.0)
    f["tracked_for"] = np.where(~raw_missing, seconds_since(raw_missing), 0.0)
    f["tracked_until"] = np.where(~raw_missing, seconds_until(raw_missing), 0.0)
    f["missing_past2"] = roll(raw_missing, 2, "mean", "past")
    f["missing_next2"] = roll(raw_missing, 2, "mean", "future")
    f["ball_abs_x"] = np.abs(bx)
    f["ball_abs_y"] = np.abs(by)
    f["ball_x_home_attack"] = bx * direction
    f["to_goal_line"] = HALF_L - np.abs(bx)
    f["to_touch_line"] = HALF_W - np.abs(by)
    f["out_of_lines"] = np.where(missing, np.nan, out_of_lines.astype(float))
    f["out_past2"] = roll(np.where(missing, np.nan, out_of_lines), 2, "mean", "past")
    f["out_next2"] = roll(np.where(missing, np.nan, out_of_lines), 2, "mean", "future")
    f["ball_speed"] = bspeed
    f["ball_speed_past1"] = roll(bspeed, 1, "mean", "past")
    f["ball_speed_next1"] = roll(bspeed, 1, "mean", "future")
    f["ball_dv"] = dv
    f["ball_dv_past1"] = roll(dv, 1, "max", "past")
    f["ball_dv_next1"] = roll(dv, 1, "max", "future")
    f["ball_still_for"] = np.where(still, seconds_since(~still), 0.0)

    # who is near the ball
    for tag, p in (("h", ph), ("a", pa)):
        for key, val in p.items():
            f[f"{key}_{tag}"] = val
        close = np.nan_to_num((p["d1"] < ON_BALL_M).astype(float))
        control = np.nan_to_num(((p["d1"] < ON_BALL_M) & (p["rel"] < 2.5)).astype(float))
        f[f"close_{tag}_past2"] = roll(close, 2, "mean", "past")
        f[f"close_{tag}_next2"] = roll(close, 2, "mean", "future")
        f[f"control_{tag}_past1"] = roll(control, 1, "mean", "past")
        f[f"control_{tag}_next1"] = roll(control, 1, "mean", "future")
        f[f"control_{tag}_past4"] = roll(control, 4, "mean", "past")
        f[f"control_{tag}_next4"] = roll(control, 4, "mean", "future")
        f[f"d1_{tag}_min_past1"] = roll(p["d1"], 1, "min", "past")
        f[f"d1_{tag}_min_next1"] = roll(p["d1"], 1, "min", "future")
    gap = np.clip(ph["d1"] - pa["d1"], -30, 30)
    f["d1_gap"] = gap
    for s in (1, 3):
        f[f"d1_gap_past{s}"] = roll(gap, s, "mean", "past")
        f[f"d1_gap_next{s}"] = roll(gap, s, "mean", "future")

    # touches
    f.update(_touch_context(touches, n))

    # team movement and shape
    spd_h, spd_a = np.nanmean(Hs, axis=1), np.nanmean(As, axis=1)
    f["speed_h"], f["speed_a"] = spd_h, spd_a
    f["speed_all_past3"] = roll((spd_h + spd_a) / 2, 3, "mean", "past")
    f["speed_all_next3"] = roll((spd_h + spd_a) / 2, 3, "mean", "future")
    cx_h, cx_a = np.nanmean(H[..., 0], axis=1), np.nanmean(A[..., 0], axis=1)
    f["centroid_h_adv"] = cx_h * direction
    f["centroid_a_adv"] = -cx_a * direction
    f["ball_vs_centroid_h"] = (bx - cx_h) * direction
    f["ball_vs_centroid_a"] = -(bx - cx_a) * direction
    f["spread_x_h"], f["spread_y_h"] = np.nanstd(H[..., 0], axis=1), np.nanstd(H[..., 1], axis=1)
    f["spread_x_a"], f["spread_y_a"] = np.nanstd(A[..., 0], axis=1), np.nanstd(A[..., 1], axis=1)
    f["players_h"] = np.isfinite(H[..., 0]).sum(axis=1).astype(float)
    f["players_a"] = np.isfinite(A[..., 0]).sum(axis=1).astype(float)

    feats = pd.DataFrame(f)
    feats.insert(0, "frame", m.frame)
    feats.insert(1, "period", m.period)
    return feats, touches


FEATURE_EXCLUDE = ("frame", "period")


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in FEATURE_EXCLUDE]

"""Ground-truth possession built from the Metrica event log.

The rules are deliberately simple:

* An on-ball event (pass, shot, recovery, ball lost, set piece) gives possession
  to the event's team from its start frame.
* The team keeps it until the other team's next on-ball event, or until the
  ball goes out of play.
* The ball goes dead when it crosses a line (end frame of BALL OUT, or of a
  shot that ends in a goal or out), when the referee whistles (FAULT RECEIVED,
  a challenge tagged FAULT, an offside) and when a half ends. It stays dead
  until the next on-ball event, which in practice is the restart.
* A whistle only counts if a set piece follows it; otherwise the referee
  played advantage and the ball never went dead.
* Goal kicks are logged as a goalkeeper RECOVERY on the same frame as a
  "GOAL KICK" pass. When the recovery comes earlier than that pass, play
  restarts at the pass.
* Every period starts dead and comes alive at the kick-off.
* Metrica only tracks the ball while it is in play, so a stretch of more than
  five seconds with no ball and no logged on-ball event is treated as a
  stoppage the event log missed (injuries, drop balls).

Frames between the kick-off and the last logged event of each period are the
ones we evaluate on; the minutes after the final whistle are left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Match

HOME, AWAY, DEAD = 0, 1, 2
CLASS_NAMES = ("home", "away", "dead")
ON_BALL = ("PASS", "BALL LOST", "RECOVERY", "SET PIECE", "SHOT")
STOPPAGE_GAP_S = 5.0


def m_fps(m: Match) -> float:
    return round(1.0 / np.median(np.diff(m.time)))


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _ball_goes_dead(ev) -> int | None:
    """Frame at which this event stops play, or None."""
    sub = ev.subtype
    if ev.type == "BALL OUT":
        return int(ev.end_frame)
    if ev.type == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return int(ev.end_frame)
    if "END HALF" in sub:
        return int(ev.end_frame)
    if _is_whistle(ev):
        return int(ev.start_frame)
    return None


def _is_whistle(ev) -> bool:
    """Fouls and offsides. They stop play only if a set piece follows."""
    return (ev.type == "FAULT RECEIVED"
            or (ev.type == "CHALLENGE" and "FAULT" in ev.subtype)
            or (ev.type == "BALL LOST" and "OFFSIDE" in ev.subtype))


def possession_labels(m: Match) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, eval_mask), one value per tracking frame."""
    n = len(m)
    state = np.full(n, -1, dtype=np.int16)

    # Each period opens with a dead ball.
    period_start = np.flatnonzero(np.r_[True, m.period[1:] != m.period[:-1]])
    state[period_start] = DEAD

    events = m.events.sort_values("start_frame", kind="stable").reset_index(drop=True)
    on_ball = events.type.isin(ON_BALL).to_numpy()
    on_ball_idx = np.flatnonzero(on_ball)

    def next_on_ball(i: int):
        j = np.searchsorted(on_ball_idx, i, side="right")
        return events.iloc[on_ball_idx[j]] if j < len(on_ball_idx) else None

    changes = []
    dead = False
    for i, ev in enumerate(events.itertuples(index=False)):
        if ev.type in ON_BALL:
            team = HOME if ev.team == "Home" else AWAY
            start = int(ev.start_frame)
            if dead and ev.type == "RECOVERY":
                nxt = next_on_ball(i)
                if nxt is not None and "GOAL KICK" in nxt.subtype and nxt.team == ev.team:
                    start = int(nxt.start_frame)
            changes.append((start, 0, team))
            dead = False
        stop = _ball_goes_dead(ev)
        if stop is not None and _is_whistle(ev):
            nxt = next_on_ball(i)
            if nxt is None or nxt.type != "SET PIECE":
                stop = None  # advantage played, the ball never went dead
        if stop is not None:
            changes.append((stop, 1, DEAD))
            dead = True
    # Sorting by (frame, order) means a dead-ball change wins over an on-ball
    # event logged on the same frame (e.g. the "END HALF" ball-lost events).
    for frame, _, value in sorted(changes):
        i = frame - m.frame[0]
        if 0 <= i < n:
            state[i] = value

    labels = pd.Series(state).replace(-1, np.nan).ffill().to_numpy().astype(np.int8)

    # Metrica only tracks the ball while it is in play. A long stretch with no
    # ball and no on-ball event is a stoppage the event log did not record
    # (injuries, drop balls), so we call it dead.
    min_gap = int(STOPPAGE_GAP_S * m_fps(m))
    missing = np.isnan(m.ball[:, 0]) & (labels != DEAD)
    starts, ends = _runs(missing)
    event_idx = np.sort(events.loc[on_ball, "start_frame"].to_numpy(int) - m.frame[0])
    for s, e in zip(starts, ends):
        if e - s < min_gap:
            continue
        k = np.searchsorted(event_idx, s)
        if k < len(event_idx) and event_idx[k] < e:
            continue  # play was logged inside the gap
        labels[s:e] = DEAD

    mask = np.zeros(n, dtype=bool)
    ev = m.events
    for p in np.unique(m.period):
        ep = ev[ev.period == p]
        if ep.empty:
            continue
        first = int(ep[ep.type.isin(ON_BALL)].start_frame.min())
        last = int(max(ep.start_frame.max(), ep.end_frame.max()))
        lo, hi = first - m.frame[0], last - m.frame[0]
        mask[max(lo, 0): min(hi + 1, n)] = True
    mask &= np.isin(m.period, (1, 2))
    return labels, mask

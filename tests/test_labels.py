import numpy as np
import pandas as pd

from possession.data import Match
from possession.labels import AWAY, DEAD, HOME, possession_labels

COLUMNS = ["team", "type", "subtype", "period", "start_frame", "end_frame"]


def make_match(n, events, ball_missing=()):
    ball = np.zeros((n, 2))
    for a, b in ball_missing:
        ball[a:b] = np.nan
    ev = pd.DataFrame(events, columns=COLUMNS)
    ev["from_player"] = ""
    ev["to_player"] = ""
    frame = np.arange(1, n + 1)
    return Match(game=0, period=np.ones(n, dtype=int), frame=frame, time=frame / 25,
                 home=np.zeros((n, 1, 2)), away=np.zeros((n, 1, 2)), ball=ball,
                 home_ids=["1"], away_ids=["2"], events=ev)


def test_possession_and_ball_out():
    events = [
        ("Home", "SET PIECE", "KICK OFF", 1, 11, 0),
        ("Home", "PASS", "", 1, 11, 20),
        ("Away", "RECOVERY", "INTERCEPTION", 1, 31, 31),
        ("Away", "BALL OUT", "", 1, 41, 51),
        ("Home", "SET PIECE", "THROW IN", 1, 81, 81),
        ("Home", "PASS", "", 1, 81, 90),
    ]
    y, mask = possession_labels(make_match(100, events))
    assert (y[:10] == DEAD).all()        # before the kick-off
    assert (y[10:30] == HOME).all()
    assert (y[30:50] == AWAY).all()      # until the ball crosses the line on frame 51
    assert (y[50:80] == DEAD).all()
    assert (y[80:] == HOME).all()
    assert mask[10] and not mask[5]


def test_foul_with_advantage_does_not_stop_play():
    events = [
        ("Home", "SET PIECE", "KICK OFF", 1, 1, 0),
        ("Away", "FAULT RECEIVED", "", 1, 21, 21),
        ("Away", "PASS", "", 1, 25, 30),           # play went on
        ("Away", "FAULT RECEIVED", "", 1, 51, 51),
        ("Away", "SET PIECE", "FREE KICK", 1, 71, 71),
    ]
    y, _ = possession_labels(make_match(90, events))
    assert (y[20:24] == HOME).all()
    assert (y[24:50] == AWAY).all()
    assert (y[50:70] == DEAD).all()
    assert (y[70:] == AWAY).all()


def test_long_untracked_gap_without_events_is_dead():
    events = [("Home", "SET PIECE", "KICK OFF", 1, 1, 0), ("Home", "PASS", "", 1, 300, 310)]
    y, _ = possession_labels(make_match(400, events, ball_missing=[(100, 250)]))  # 6 s
    assert (y[100:250] == DEAD).all()
    assert (y[:100] == HOME).all() and (y[250:299] == HOME).all()


def test_short_untracked_gap_stays_live():
    events = [("Home", "SET PIECE", "KICK OFF", 1, 1, 0), ("Home", "PASS", "", 1, 300, 310)]
    y, _ = possession_labels(make_match(400, events, ball_missing=[(100, 150)]))  # 2 s
    assert (y[100:150] == HOME).all()

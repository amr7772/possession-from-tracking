import math

import numpy as np

from possession.video import (NONE, NO_VIEW_SHARE, OTHER, TEAM_A, TEAM_B, Person, Sample, clean_ball, colour_name,
                              emission, gap_lengths, rect_distance, smoother_params, spells, team_distances)

PRIOR = np.array([0.29, 0.29, 0.42])


def person(x, y, team, h=40):
    return Person(box=np.array([x - 10, y - h, x + 10, y], np.float32), conf=0.9, kind="person",
                  colour=None, team=team)


def test_rect_distance_inside_and_outside():
    box = np.array([0, 0, 10, 20])
    assert rect_distance(5, 5, box) == 0
    assert rect_distance(13, 24, box) == 5


def test_team_distances_in_player_heights():
    people = [person(100, 200, TEAM_A), person(300, 200, TEAM_B), person(110, 210, OTHER)]
    d_a, d_b = team_distances(np.array([100, 220]), people)
    assert math.isclose(d_a, 0.5)
    assert d_b > 4


def test_emission_is_neutral_when_nobody_is_near():
    far = emission(math.inf, math.inf, PRIOR)
    assert np.allclose(far, PRIOR / PRIOR.sum())
    near_a = emission(0.1, 3.0, PRIOR)
    assert near_a.argmax() == TEAM_A and near_a[TEAM_A] > 0.7


def test_clean_ball_drops_spike_and_fills_gap():
    samples = [Sample(frame=i, t=i / 5, pitch_share=0.8) for i in range(8)]
    for i, x in enumerate([100, 110, 1500, 130, None, None, 160, 170]):
        if x is not None:
            samples[i].ball = np.array([x, 300], np.float32)
    pos, source = clean_ball(samples, width=1920, hz=5)
    assert source[2] == 2 and abs(pos[2, 0] - 120) < 1e-3       # the jump to 1500 is gone, refilled
    assert list(source[4:6]) == [2, 2] and abs(pos[5, 0] - 150) < 1e-3


def test_clean_ball_ignores_frames_without_pitch():
    s = Sample(frame=0, t=0, pitch_share=NO_VIEW_SHARE / 2, ball=np.array([5, 5], np.float32))
    pos, source = clean_ball([s], width=100, hz=5)
    assert source[0] == 0 and np.isnan(pos[0, 0])


def test_gap_lengths():
    known = np.array([True, False, False, True, False])
    assert np.allclose(gap_lengths(known, 0.2), [0, 0.4, 0.4, 0, 0.2])


def test_spells_cover_the_whole_time():
    states = np.array([TEAM_A, TEAM_A, NONE, TEAM_B, TEAM_B, TEAM_B])
    times = np.arange(6) * 0.2
    runs = spells(states, times, 0.2)
    assert [r["state"] for r in runs] == ["team_a", "none", "team_b"]
    assert math.isclose(sum(r["seconds"] for r in runs), 1.2)


def test_smoother_is_symmetric_between_teams():
    prior, trans, _ = smoother_params(None)
    assert math.isclose(prior[TEAM_A], prior[TEAM_B])
    assert math.isclose(trans[TEAM_A, TEAM_B], trans[TEAM_B, TEAM_A])
    assert np.allclose(trans.sum(axis=1), 1)


def test_colour_names():
    assert colour_name("#f4f4f4") == "white"
    assert colour_name("#111111") == "black"
    assert colour_name("#c8102e") == "red"
    assert colour_name("#acec8f") == "green"
    assert colour_name("#1d428a") == "blue"


def test_parse_teams():
    from possession.video import parse_teams
    assert parse_teams("Zamalek=white, Al Ahly=Red") == [("Zamalek", "white"), ("Al Ahly", "red")]
    assert parse_teams("A,B") == [("A", None), ("B", None)]
    assert parse_teams("") is None


def test_match_names_pairs_shirts_with_seen_colours():
    from possession.video import match_names
    seen = ["#acec8f", "#c6d0d2"]                       # lime shirts look lighter than shaded white
    assert match_names(seen, [("Gladbach", "white"), ("Wolfsburg", "green")]) == ["Wolfsburg", "Gladbach"]
    assert match_names(seen, [("Wolfsburg", "lime"), ("Gladbach", "#ffffff")]) == ["Wolfsburg", "Gladbach"]
    assert match_names(seen, [("First", None), ("Second", None)]) == ["First", "Second"]

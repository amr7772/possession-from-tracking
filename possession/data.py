"""Reading the Metrica Sports sample data (games 1 and 2, CSV format).

Coordinates come normalised to [0, 1] with (0, 0) at the top-left corner.
We convert them to metres with the origin on the centre spot:
x runs from -52.5 to 52.5 (left to right) and y from -34 to 34 (top to bottom).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FPS = 25

_FILES = ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")


def download_game(game: int, root: Path | str) -> Path:
    """Fetch the three CSV files of a sample game unless they are already there."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        fname = f"Sample_Game_{game}_{name}.csv"
        dest = root / fname
        if dest.exists():
            continue
        url = f"{BASE_URL}/Sample_Game_{game}/{fname}"
        print(f"  downloading {fname}")
        urllib.request.urlretrieve(url, dest)
    return root


def to_metres(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x - 0.5) * PITCH_LENGTH, (y - 0.5) * PITCH_WIDTH


def _read_tracking(path: Path, team: str) -> tuple[pd.DataFrame, list[str]]:
    # Rows 0 and 1 hold team names and shirt numbers; row 2 is the real header,
    # with one named column per player followed by an unnamed one for y.
    df = pd.read_csv(path, skiprows=2)
    names = list(df.columns)
    cols = ["period", "frame", "time"]
    ids: list[str] = []
    for i in range(3, len(names), 2):
        name = names[i]
        if name.startswith("Player"):
            pid = name[len("Player"):]
            ids.append(pid)
            cols += [f"{team}_{pid}_x", f"{team}_{pid}_y"]
        elif name == "Ball":
            cols += ["ball_x", "ball_y"]
        else:
            raise ValueError(f"unexpected column {name!r} in {path.name}")
    df.columns = cols
    return df, ids


def load_events(path: Path | str) -> pd.DataFrame:
    e = pd.read_csv(path)
    e = e.rename(columns={
        "Team": "team", "Type": "type", "Subtype": "subtype", "Period": "period",
        "Start Frame": "start_frame", "Start Time [s]": "start_time",
        "End Frame": "end_frame", "End Time [s]": "end_time",
        "From": "from_player", "To": "to_player",
        "Start X": "start_x", "Start Y": "start_y", "End X": "end_x", "End Y": "end_y",
    })
    for col in ("subtype", "from_player", "to_player"):
        e[col] = e[col].fillna("").astype(str)
    return e


@dataclass
class Match:
    game: int
    period: np.ndarray      # (T,)
    frame: np.ndarray       # (T,) Metrica frame numbers, contiguous from 1
    time: np.ndarray        # (T,) seconds since the first frame
    home: np.ndarray        # (T, n_home, 2) metres, NaN while a player is off the pitch
    away: np.ndarray        # (T, n_away, 2)
    ball: np.ndarray        # (T, 2) metres, NaN while the ball is not tracked
    home_ids: list[str]
    away_ids: list[str]
    events: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    def index_of(self, frame: int | np.ndarray) -> int | np.ndarray:
        return np.asarray(frame) - self.frame[0]


def _stack(df: pd.DataFrame, team: str, ids: list[str]) -> np.ndarray:
    xs = np.stack([df[f"{team}_{p}_x"].to_numpy(float) for p in ids], axis=1)
    ys = np.stack([df[f"{team}_{p}_y"].to_numpy(float) for p in ids], axis=1)
    x, y = to_metres(xs, ys)
    return np.stack([x, y], axis=2)


def load_game(game: int, root: Path | str) -> Match:
    root = Path(root)
    home, home_ids = _read_tracking(root / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", "home")
    away, away_ids = _read_tracking(root / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", "away")
    frame = home["frame"].to_numpy(int)
    if not np.array_equal(frame, away["frame"].to_numpy(int)):
        raise ValueError("home and away tracking files are not aligned")
    if not np.array_equal(np.diff(frame), np.ones(len(frame) - 1, dtype=int)):
        raise ValueError("expected contiguous frame numbers")
    bx, by = to_metres(home["ball_x"].to_numpy(float), home["ball_y"].to_numpy(float))
    return Match(
        game=game,
        period=home["period"].to_numpy(int),
        frame=frame,
        time=home["time"].to_numpy(float),
        home=_stack(home, "home", home_ids),
        away=_stack(away, "away", away_ids),
        ball=np.stack([bx, by], axis=1),
        home_ids=home_ids,
        away_ids=away_ids,
        events=load_events(root / f"Sample_Game_{game}_RawEventsData.csv"),
    )

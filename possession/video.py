"""Possession from match video.

Reads a football video and says, five times a second, which team has the ball.
It needs no tracking data: the players and the ball are found in the picture.

    python -m possession.video match.mp4 --out out/

For every sampled frame (5 a second by default):

1. Find people and the ball with a YOLO detector. A COCO model works out of the
   box; a football model with player / goalkeeper / referee / ball classes is
   used the same way if you pass it with --model.
2. Keep the people standing on the grass. A green mask of the pitch drops the
   crowd and most of the touchline.
3. Split the players into two teams by shirt colour. The colours of the whole
   video are clustered once, so lit and shadowed shirts of one team stay together.
4. Measure how far the ball is from the nearest player of each team, in player
   heights, so the measure means the same near and far from the camera.
5. Turn the distances into probabilities and smooth them with the hidden Markov
   model of the tracking-data model (possession/smoothing.py). Its transition
   matrix was learned from a real match at 5 Hz, so it knows how often
   possession changes hands.

Writes possession.csv, summary.json, report.html and annotated.mp4 into --out,
plus detections.npz, which a second run with the same video and settings reuses
(to rename the teams, say) instead of running the detector again.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .smoothing import decode, make_sticky

TEAM_A, TEAM_B, NONE = 0, 1, 2
OTHER = -1                      # referees, goalkeepers, staff: never given the ball
STATE_NAMES = ("team_a", "team_b", "none")

PERSON_MIN_CONF = 0.30
BALL_MIN_CONF = 0.05            # low on purpose: the ball is small; the pitch mask and
                                # the spike filter below clean up the false hits
NO_VIEW_SHARE = 0.20            # less grass than this in the frame: close-up, crowd or replay
TAU = 0.6                       # distance (in player heights) at which a team's closeness falls to 1/e
FLOOR = 0.1                     # score each team keeps when nobody is near the ball, so a pass in
                                # flight says nothing and possession stays with the passer's team
LONG_GAP_S = 2.0                # ball unseen for longer than this: start leaning to "nobody"
LOST_BALL = np.array([0.15, 0.15, 0.7])
NO_VIEW = np.array([0.05, 0.05, 0.9])


@dataclass
class Person:
    box: np.ndarray             # x1, y1, x2, y2 in pixels of the original frame
    conf: float
    kind: str                   # player, goalkeeper, referee or person (COCO)
    colour: np.ndarray | None   # median shirt colour in OpenCV Lab
    team: int = OTHER


@dataclass
class Sample:
    frame: int
    t: float
    pitch_share: float
    people: list[Person] = field(default_factory=list)
    ball: np.ndarray | None = None      # x, y in pixels
    ball_conf: float = 0.0


# --------------------------------------------------------------------------- video


def probe(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def sample_frames(path: Path, hz: float, start: float = 0.0, duration: float | None = None):
    """Yield (frame number, seconds, BGR image) about hz times a second of video."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / hz))
    first = int(round(start * fps))
    last = None if duration is None else first + int(round(duration * fps))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    i = first
    while last is None or i < last:
        if (i - first) % step == 0:
            ok, img = cap.read()
            if not ok:
                break
            yield i, i / fps, img
        elif not cap.grab():
            break
        i += 1
    cap.release()


# --------------------------------------------------------------------------- pitch


def pitch_mask(img: np.ndarray, width: int = 320) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Mask of the playing surface on a small copy of the frame.

    Returns the filled mask, the share of the frame it covers, the scale from the
    original frame to the mask, and the median Lab colour of the grass.
    """
    import cv2
    h, w = img.shape[:2]
    scale = width / w
    small = cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n <= 1:
        return np.zeros_like(closed), 0.0, scale, np.array([0.0, 0.0, 0.0])
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == biggest).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    grass_px = lab[(green.reshape(-1) > 0) & (filled.reshape(-1) > 0)]
    grass = np.median(grass_px, axis=0).astype(np.float32) if len(grass_px) else np.zeros(3, np.float32)
    return filled, float(filled.mean() / 255), scale, grass


def on_mask(mask: np.ndarray, scale: float, x: float, y: float) -> bool:
    mx, my = int(x * scale), int(y * scale)
    mh, mw = mask.shape
    return 0 <= mx < mw and 0 <= my < mh and mask[my, mx] > 0


def shirt_colour(img: np.ndarray, box: np.ndarray, grass: np.ndarray) -> np.ndarray | None:
    """Median Lab colour of the shirt: the middle of the upper body, without grass pixels."""
    import cv2
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + 0.25 * w), int(math.ceil(x2 - 0.25 * w))
    ya, yb = int(y1 + 0.15 * h), int(math.ceil(y1 + 0.50 * h))
    H, W = img.shape[:2]
    xa, xb, ya, yb = max(xa, 0), min(xb, W), max(ya, 0), min(yb, H)
    if xb <= xa or yb <= ya:
        return None
    lab = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    far = np.linalg.norm(lab - grass, axis=1) > 20
    if far.sum() >= max(4, 0.2 * len(lab)):
        lab = lab[far]
    return np.median(lab, axis=0)


# --------------------------------------------------------------------------- detection


def load_detector(weights: str):
    """A YOLO model and the ids of its person-like and ball classes."""
    from ultralytics import YOLO
    model = YOLO(weights)
    names = {int(k): v.lower() for k, v in model.names.items()}
    football = {k: v for k, v in names.items() if v in ("player", "goalkeeper", "referee")}
    people = football or {k: v for k, v in names.items() if v == "person"}
    balls = {k: v for k, v in names.items() if v in ("ball", "sports ball")}
    if not people or not balls:
        raise SystemExit(f"{weights} has no person or ball class: {sorted(names.values())}")
    return model, people, balls


def detect(path: Path, weights: str, hz: float, imgsz: int, device: str, batch: int = 8,
           start: float = 0.0, duration: float | None = None, log=print) -> list[Sample]:
    model, people_ids, ball_ids = load_detector(weights)
    classes = sorted(people_ids) + sorted(ball_ids)
    info = probe(path)
    total = info["frames"] / info["fps"] if info["frames"] > 0 else None
    if total is not None:
        total = max(0.0, min(total - start, duration or total))
    expected = int(total * hz) if total else None
    samples: list[Sample] = []
    t0 = time.time()

    def run(buf):
        results = model.predict([b[2] for b in buf], imgsz=imgsz, conf=BALL_MIN_CONF, classes=classes,
                                device=device, verbose=False)
        for (frame, t, img), r in zip(buf, results):
            mask, share, scale, grass = pitch_mask(img)
            s = Sample(frame=frame, t=t, pitch_share=share)
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            best = -1.0
            for box, c, p in zip(xyxy, cls, conf):
                if c in people_ids:
                    feet_x, feet_y = (box[0] + box[2]) / 2, box[3]
                    if p < PERSON_MIN_CONF or not on_mask(mask, scale, feet_x, feet_y):
                        continue
                    s.people.append(Person(box=box.astype(np.float32), conf=float(p), kind=people_ids[c],
                                           colour=shirt_colour(img, box, grass)))
                elif c in ball_ids and p > best:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if on_mask(mask, scale, cx, cy):
                        best = float(p)
                        s.ball, s.ball_conf = np.array([cx, cy], np.float32), best
            samples.append(s)

    buf = []
    for item in sample_frames(path, hz, start, duration):
        buf.append(item)
        if len(buf) == batch:
            run(buf)
            buf = []
            n = len(samples)
            if n % (batch * 25) == 0:
                rate = n / (time.time() - t0)
                left = f", about {(expected - n) / rate / 60:.0f} min left" if expected else ""
                log(f"  {n} frames analysed ({samples[-1].t / 60:.1f} min of video){left}")
    if buf:
        run(buf)
    return samples


KINDS = ("player", "goalkeeper", "referee", "person")


def save_detections(path: Path, samples: list[Sample], key: dict) -> None:
    """Detections in plain arrays, so a second run (new team names, other settings) skips the detector."""
    rows = [(i, p) for i, s in enumerate(samples) for p in s.people]
    nan3 = np.full(3, np.nan, np.float32)
    np.savez_compressed(
        path, key=np.array(json.dumps(key, sort_keys=True)),
        frame=np.array([s.frame for s in samples], np.int64), t=np.array([s.t for s in samples]),
        share=np.array([s.pitch_share for s in samples], np.float32),
        ball=np.array([s.ball if s.ball is not None else (np.nan, np.nan) for s in samples], np.float32).reshape(-1, 2),
        ball_conf=np.array([s.ball_conf for s in samples], np.float32),
        p_sample=np.array([i for i, _ in rows], np.int64),
        p_box=np.array([p.box for _, p in rows], np.float32).reshape(-1, 4),
        p_conf=np.array([p.conf for _, p in rows], np.float32),
        p_kind=np.array([KINDS.index(p.kind) for _, p in rows], np.int8),
        p_colour=np.array([p.colour if p.colour is not None else nan3 for _, p in rows], np.float32).reshape(-1, 3))


def load_detections(path: Path, key: dict) -> list[Sample] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["key"]) != json.dumps(key, sort_keys=True):
            return None
        samples = [Sample(frame=int(f), t=float(t), pitch_share=float(sh),
                          ball=None if np.isnan(b[0]) else b.copy(), ball_conf=float(bc))
                   for f, t, sh, b, bc in zip(z["frame"], z["t"], z["share"], z["ball"], z["ball_conf"])]
        for i, box, conf, kind, colour in zip(z["p_sample"], z["p_box"], z["p_conf"], z["p_kind"], z["p_colour"]):
            samples[i].people.append(Person(box=box.copy(), conf=float(conf), kind=KINDS[int(kind)],
                                            colour=None if np.isnan(colour[0]) else colour.copy()))
    return samples


# --------------------------------------------------------------------------- teams


def _weighted(lab: np.ndarray) -> np.ndarray:
    """Lab with lightness counted half, so sun and shadow on one shirt stay close."""
    return lab * np.array([0.5, 1.0, 1.0], np.float32)


def colour_hex(lab: np.ndarray) -> str:
    import cv2
    px = np.clip(np.round(lab), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def colour_name(hex_colour: str) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hi, lo = max(r, g, b), min(r, g, b)
    sat = 0 if hi == 0 else (hi - lo) / hi
    if hi < 0.25:
        return "black"
    if sat < 0.18:
        return "white" if hi > 0.7 else "grey"
    if hi == r:
        hue = (60 * ((g - b) / (hi - lo))) % 360
    elif hi == g:
        hue = 60 * ((b - r) / (hi - lo)) + 120
    else:
        hue = 60 * ((r - g) / (hi - lo)) + 240
    for limit, name in ((15, "red"), (40, "orange"), (65, "yellow"), (165, "green"), (195, "cyan"),
                        (255, "blue"), (290, "purple"), (335, "pink"), (360, "red")):
        if hue < limit:
            return name
    return "red"


COLOUR_WORDS = {
    "white": "#f2f2f2", "black": "#151515", "grey": "#808080", "gray": "#808080", "silver": "#c0c0c0",
    "red": "#d0202e", "maroon": "#7a1f2b", "claret": "#7a1f3d", "orange": "#f08a24", "yellow": "#f5d90a",
    "gold": "#d4a72c", "green": "#2e9e44", "lime": "#a6e22e", "cyan": "#33c3e0", "sky": "#6ab7e8",
    "blue": "#1f4fbf", "navy": "#1b2a4e", "purple": "#6f3a9a", "violet": "#7f3fbf", "pink": "#f06ba8",
    "brown": "#7b4a2a",
}


def _lab(hex_colour: str) -> np.ndarray:
    """sRGB hex to CIE Lab (D65)."""
    rgb = np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def parse_teams(spec: str | None) -> list[tuple[str, str | None]] | None:
    """'Zamalek=white,Al Ahly=red' -> [('Zamalek', 'white'), ('Al Ahly', 'red')]. Colours are optional."""
    if not spec or not spec.strip():
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError('Give two teams, for example "Zamalek=white,Al Ahly=red".')
    out = []
    for part in parts:
        name, _, colour = part.partition("=")
        colour = colour.strip().lower() or None
        if colour and colour not in COLOUR_WORDS and not (colour.startswith("#") and len(colour) == 7):
            raise ValueError(f"Unknown colour '{colour}'. Use a word ({', '.join(sorted(COLOUR_WORDS))}) "
                             "or a hex code such as #ffffff.")
        out.append((name.strip(), colour))
    return out


def match_names(seen_hex: list[str], teams: list[tuple[str, str | None]]) -> list[str]:
    """Names in the order of the detected teams, pairing the given shirt colours with the seen ones."""
    if any(c is None for _, c in teams):
        return [n for n, _ in teams]
    ref = [_lab(COLOUR_WORDS.get(c, c)) for _, c in teams]
    seen = [_lab(h) for h in seen_hex]
    keep = np.linalg.norm(seen[0] - ref[0]) + np.linalg.norm(seen[1] - ref[1])
    swap = np.linalg.norm(seen[0] - ref[1]) + np.linalg.norm(seen[1] - ref[0])
    return [teams[0][0], teams[1][0]] if keep <= swap else [teams[1][0], teams[0][0]]


@dataclass
class Teams:
    centres: np.ndarray         # fine k-means centres (weighted Lab)
    groups: np.ndarray          # team of each fine centre: TEAM_A, TEAM_B or OTHER
    colours: list[str]          # display colour of team A and B
    names: list[str]
    warning: str = ""

    def assign(self, colour: np.ndarray | None) -> int:
        if colour is None:
            return OTHER
        d = np.linalg.norm(self.centres - _weighted(colour), axis=1)
        return int(self.groups[int(np.argmin(d))])


def fit_teams(samples: list[Sample], given: list[tuple[str, str | None]] | None = None, merge: float = 32.0,
              seed: int = 0) -> Teams:
    """Two teams from the shirt colours of everyone on the pitch, over the whole video.

    K-means finds up to eight colour modes; modes closer than merge are joined
    (a shirt in sun and in shade); the two biggest joined groups are the teams.
    Smaller groups are referees, goalkeepers and staff. given names the teams
    (with their shirt colours, if known; see parse_teams).
    """
    from sklearn.cluster import KMeans
    pool = [p for s in samples for p in s.people if p.colour is not None and p.kind in ("player", "person")]
    if len(pool) < 20:
        raise SystemExit("Too few players found to tell the teams apart. Is this a match video?")
    lab = np.array([p.colour for p in pool], np.float32)
    rng = np.random.default_rng(seed)
    if len(lab) > 20000:
        lab = lab[rng.choice(len(lab), 20000, replace=False)]
    X = _weighted(lab)
    k = min(8, len(X) // 5)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    centres = km.cluster_centers_.astype(np.float32)
    sizes = np.bincount(km.labels_, minlength=k)

    parent = list(range(k))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            if np.linalg.norm(centres[i] - centres[j]) < merge:
                parent[root(i)] = root(j)
    roots = np.array([root(i) for i in range(k)])
    group_size = {r: sizes[roots == r].sum() for r in set(roots)}
    ranked = sorted(group_size, key=group_size.get, reverse=True)
    if len(ranked) < 2:
        raise SystemExit("All shirts look alike; cannot split the players into two teams.")
    team_roots = ranked[:2]

    def mean_lab(r):
        sel = roots == r
        w = sizes[sel][:, None]
        c = (centres[sel] * w).sum(0) / w.sum()
        return c / np.array([0.5, 1.0, 1.0], np.float32)

    team_lab = [mean_lab(r) for r in team_roots]
    if team_lab[1][0] > team_lab[0][0]:            # lighter shirt is team A, so runs are repeatable
        team_roots, team_lab = team_roots[::-1], team_lab[::-1]
    groups = np.full(k, OTHER)
    groups[roots == team_roots[0]] = TEAM_A
    groups[roots == team_roots[1]] = TEAM_B
    colours = [colour_hex(c) for c in team_lab]
    if given:
        names = match_names(colours, given)
    else:
        a, b = colour_name(colours[0]), colour_name(colours[1])
        names = [f"{a.capitalize()} team", f"{b.capitalize()} team"] if a != b else ["Team A", "Team B"]
    gap = np.linalg.norm(_weighted(team_lab[0]) - _weighted(team_lab[1]))
    small = group_size[team_roots[1]] / max(1, group_size[team_roots[0]])
    warning = ""
    if gap < 1.5 * merge:
        warning = "The two kits are close in colour; check the team split in annotated.mp4."
    elif small < 0.35:
        warning = "One team was seen far less than the other; check the team split in annotated.mp4."
    return Teams(centres=centres, groups=groups, colours=colours, names=list(names), warning=warning)


# --------------------------------------------------------------------------- ball


def clean_ball(samples: list[Sample], width: int, hz: float, max_gap_s: float = 1.0,
               jump: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Ball position per sample after dropping one-frame jumps and filling short gaps.

    Returns positions (n, 2) with NaN where unknown, and the source of each value:
    1 detected, 2 interpolated, 0 missing.
    """
    n = len(samples)
    pos = np.full((n, 2), np.nan, np.float32)
    for i, s in enumerate(samples):
        if s.ball is not None and s.pitch_share >= NO_VIEW_SHARE:
            pos[i] = s.ball
    limit = jump * width
    seen = np.flatnonzero(~np.isnan(pos[:, 0]))
    for a, b, c in zip(seen, seen[1:], seen[2:]):
        if c - a <= 4:
            out_in = np.linalg.norm(pos[b] - pos[a]) > limit and np.linalg.norm(pos[c] - pos[b]) > limit
            if out_in and np.linalg.norm(pos[c] - pos[a]) < limit:
                pos[b] = np.nan
    source = np.where(np.isnan(pos[:, 0]), 0, 1).astype(np.int8)
    seen = np.flatnonzero(source == 1)
    max_gap = max(1, int(round(max_gap_s * hz)))
    for a, b in zip(seen, seen[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            pos[a + 1:b] = pos[a] + w[:, None] * (pos[b] - pos[a])
            source[a + 1:b] = 2
    return pos, source


# --------------------------------------------------------------------------- possession


def rect_distance(x: float, y: float, box: np.ndarray) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def team_distances(ball: np.ndarray, people: list[Person]) -> tuple[float, float]:
    """Distance from the ball to the nearest player of each team, in that player's heights."""
    best = [math.inf, math.inf]
    for p in people:
        if p.team in (TEAM_A, TEAM_B):
            h = max(1.0, float(p.box[3] - p.box[1]))
            best[p.team] = min(best[p.team], rect_distance(float(ball[0]), float(ball[1]), p.box) / h)
    return best[0], best[1]


def emission(d_a: float, d_b: float, prior: np.ndarray, tau: float = TAU, floor: float = FLOOR) -> np.ndarray:
    """p(team A, team B, nobody) for a frame where the ball is visible.

    With nobody near the ball the result equals the prior, which the smoother reads
    as "no news": possession stays where it was, as it should during a pass.
    """
    none = floor * prior[2] / prior[0]
    score = np.array([floor + math.exp(-(d_a / tau) ** 2), floor + math.exp(-(d_b / tau) ** 2), none])
    return score / score.sum()


def smoother_params(config: Path | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Prior and transitions of the tracking-data model, with the two teams made symmetric."""
    trans = np.array([[0.988, 0.008, 0.004], [0.008, 0.988, 0.004], [0.002, 0.002, 0.996]])
    prior, alpha, stick = np.array([0.29, 0.29, 0.42]), 1.0, 2.0
    if config is not None and config.exists():
        s = json.loads(config.read_text())["smoother"]
        trans, prior = np.array(s["transition"]), np.array(s["prior"])
        alpha, stick = float(s["alpha"]), float(s["stickiness"])
    t = trans.copy()
    t[0, 0] = t[1, 1] = (trans[0, 0] + trans[1, 1]) / 2
    t[0, 1] = t[1, 0] = (trans[0, 1] + trans[1, 0]) / 2
    t[0, 2] = t[1, 2] = (trans[0, 2] + trans[1, 2]) / 2
    t[2, 0] = t[2, 1] = (trans[2, 0] + trans[2, 1]) / 2
    t = t / t.sum(axis=1, keepdims=True)
    p = np.array([(prior[0] + prior[1]) / 2] * 2 + [prior[2]])
    return p / p.sum(), make_sticky(t, stick), alpha


def gap_lengths(known: np.ndarray, step: float) -> np.ndarray:
    """For each sample, how long (seconds) the run of unknown ball positions around it lasts."""
    out = np.zeros(len(known))
    i = 0
    while i < len(known):
        if known[i]:
            i += 1
            continue
        j = i
        while j < len(known) and not known[j]:
            j += 1
        out[i:j] = (j - i) * step
        i = j
    return out


def possession(samples: list[Sample], ball: np.ndarray, config: Path | None,
               hz: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample probabilities (n, 3) and the smoothed state of each sample."""
    prior, trans, alpha = smoother_params(config)
    gaps = gap_lengths(~np.isnan(ball[:, 0]), 1.0 / hz)
    proba = np.empty((len(samples), 3))
    for i, s in enumerate(samples):
        if s.pitch_share < NO_VIEW_SHARE:
            proba[i] = NO_VIEW
        elif np.isnan(ball[i, 0]):
            proba[i] = LOST_BALL if gaps[i] > LONG_GAP_S else prior
        else:
            proba[i] = emission(*team_distances(ball[i], s.people), prior)
    states = decode(proba, prior, trans, alpha, np.zeros(len(samples), dtype=int))
    return proba, states


def spells(states: np.ndarray, times: np.ndarray, step: float) -> list[dict]:
    """Unbroken runs of one state, with start, end and length in seconds."""
    out = []
    if len(states) == 0:
        return out
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append({"state": STATE_NAMES[int(states[start])], "start": float(times[start]),
                        "end": float(times[i - 1] + step), "seconds": float(times[i - 1] + step - times[start])})
            start = i
    return out


def summarise(samples: list[Sample], states: np.ndarray, source: np.ndarray, teams: Teams, hz: float,
              video: Path, info: dict, weights: str) -> dict:
    times = np.array([s.t for s in samples])
    step = 1.0 / hz
    runs = spells(states, times, step)
    secs = [float((states == k).sum() * step) for k in (TEAM_A, TEAM_B, NONE)]
    held = secs[0] + secs[1]
    share = [secs[0] / held, secs[1] / held] if held else [0.0, 0.0]
    team_runs = [r for r in runs if r["state"] != "none"]
    span = float(times[-1] - times[0] + step) if len(times) else 0.0
    block = next((b for b in (10, 30, 60, 300, 900) if span / b <= 12), 900)   # 15 minutes for a full match
    blocks = []
    for b0 in np.arange(times[0], times[-1] + step, block) if len(times) else []:
        sel = (times >= b0) & (times < b0 + block)
        a, bb = int((states[sel] == TEAM_A).sum()), int((states[sel] == TEAM_B).sum())
        blocks.append({"start": float(b0), "end": float(min(b0 + block, times[-1] + step)),
                       "team_a": a / (a + bb) if a + bb else None, "team_b": bb / (a + bb) if a + bb else None})
    return {
        "video": video.name,
        "duration_seconds": span,
        "frames_analysed": len(samples),
        "rate_hz": hz,
        "detector": weights,
        "teams": [{"name": teams.names[k], "colour": teams.colours[k], "possession": share[k],
                   "seconds_on_ball": secs[k],
                   "spells": sum(r["state"] == STATE_NAMES[k] for r in team_runs),
                   "longest_spell_seconds": max([r["seconds"] for r in team_runs if r["state"] == STATE_NAMES[k]],
                                                default=0.0)}
                  for k in (TEAM_A, TEAM_B)],
        "seconds_no_possession": secs[2],
        "ball_seen_share": float((source == 1).mean()) if len(source) else 0.0,
        "ball_filled_share": float((source == 2).mean()) if len(source) else 0.0,
        "no_pitch_view_share": float(np.mean([s.pitch_share < NO_VIEW_SHARE for s in samples])) if samples else 0.0,
        "block_seconds": block,
        "blocks": blocks,
        "spells": runs,
        "warning": teams.warning,
        "video_size": [info["width"], info["height"]],
    }


# --------------------------------------------------------------------------- outputs


def write_csv(path: Path, samples: list[Sample], proba: np.ndarray, states: np.ndarray, ball: np.ndarray,
              source: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "frame", "state", "p_team_a", "p_team_b", "p_none", "ball_x", "ball_y",
                    "ball", "players_team_a", "players_team_b", "pitch_share"])
        for s, p, st, b, src in zip(samples, proba, states, ball, source):
            na = sum(q.team == TEAM_A for q in s.people)
            nb = sum(q.team == TEAM_B for q in s.people)
            w.writerow([f"{s.t:.2f}", s.frame, STATE_NAMES[int(st)], f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}",
                        "" if np.isnan(b[0]) else f"{b[0]:.1f}", "" if np.isnan(b[1]) else f"{b[1]:.1f}",
                        ("missing", "seen", "filled")[int(src)], na, nb, f"{s.pitch_share:.2f}"])


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (5, 3, 1))


def render(path: Path, out: Path, samples: list[Sample], states: np.ndarray, ball: np.ndarray, teams: Teams,
           hz: float, start: float, duration: float | None, width: int = 960) -> None:
    """Annotated video at the sampling rate: team rings at the players' feet, the ball,
    and a banner with who has the ball and the running possession split."""
    import cv2
    info = probe(path)
    scale = width / info["width"]
    size = (width, int(round(info["height"] * scale)) // 2 * 2)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
           "-r", f"{hz:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    if shutil.which("ffmpeg"):
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        write, close = proc.stdin.write, lambda: (proc.stdin.close(), proc.wait())
    else:
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), hz, size)
        write, close = (lambda b: vw.write(np.frombuffer(b, np.uint8).reshape(size[1], size[0], 3))), vw.release
    team_bgr = [_bgr(c) for c in teams.colours]
    held = np.zeros(2)
    for i, (frame, t, img) in enumerate(sample_frames(path, hz, start, duration)):
        if i >= len(samples):
            break
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        for p in samples[i].people:
            x1, y1, x2, y2 = p.box * scale
            centre = (int((x1 + x2) / 2), int(y2))
            axes = (max(6, int((x2 - x1) * 0.6)), max(3, int((x2 - x1) * 0.22)))
            if p.team in (TEAM_A, TEAM_B):
                cv2.ellipse(img, centre, axes, 0, -45, 235, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.ellipse(img, centre, axes, 0, -45, 235, team_bgr[p.team], 2, cv2.LINE_AA)
            else:                                   # referee, goalkeeper, staff: thin dark ring
                cv2.ellipse(img, centre, axes, 0, -45, 235, (40, 40, 40), 1, cv2.LINE_AA)
        st = int(states[i])
        if not np.isnan(ball[i, 0]):
            bx, by = ball[i] * scale
            tip = (int(bx), int(by) - 8)
            pts = np.array([tip, (tip[0] - 7, tip[1] - 12), (tip[0] + 7, tip[1] - 12)], np.int32)
            if st in (TEAM_A, TEAM_B):
                cv2.fillPoly(img, [pts], team_bgr[st], cv2.LINE_AA)
            cv2.polylines(img, [pts], True, (0, 0, 0) if st in (TEAM_A, TEAM_B) else (255, 255, 255), 2, cv2.LINE_AA)
        if st in (TEAM_A, TEAM_B):
            held[st] += 1
        split = held / held.sum() * 100 if held.sum() else np.array([50.0, 50.0])
        cv2.rectangle(img, (0, 0), (size[0], 34), (20, 20, 20), -1)
        x = 10
        for k in (TEAM_A, TEAM_B):
            cv2.rectangle(img, (x, 9), (x + 16, 25), team_bgr[k], -1)
            label = f"{teams.names[k]} {split[k]:.0f}%"
            cv2.putText(img, label, (x + 22, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            x += 32 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        now = teams.names[st] if st in (TEAM_A, TEAM_B) else "no possession"
        mm, ss = divmod(int(t), 60)
        right = f"Ball: {now}   {mm:02d}:{ss:02d}"
        tw = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(img, right, (size[0] - tw - 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        write(img.tobytes())
    close()


# --------------------------------------------------------------------------- main


def analyse(video: Path, out: Path, weights: str = "yolo26m.pt", hz: float = 5.0, imgsz: int = 1280,
            device: str | None = None, batch: int = 8, start: float = 0.0, duration: float | None = None,
            teams: str | None = None, make_video: bool = True, log=print) -> dict:
    """Run everything on one video. teams names the sides with their shirts: "Zamalek=white,Al Ahly=red"."""
    given = parse_teams(teams)
    if device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    log(f"{video.name}: {info['width']}x{info['height']}, {info['fps']:.0f} fps, "
        f"{info['frames'] / info['fps'] / 60:.1f} min. Looking {hz:g} times a second on {device}.")
    key = {"video": video.name, "bytes": video.stat().st_size, "weights": Path(weights).name, "hz": hz,
           "imgsz": imgsz, "start": start, "duration": duration}
    cache = out / "detections.npz"
    samples = load_detections(cache, key)
    if samples is not None:
        log("Reusing the detections saved by an earlier run with the same settings.")
    else:
        samples = detect(video, weights, hz, imgsz, device, batch, start, duration, log)
        save_detections(cache, samples, key)
    return finish(samples, video, out, info, weights, hz, start, duration, given, make_video, log)


def finish(samples: list[Sample], video: Path, out: Path, info: dict, weights: str, hz: float, start: float,
           duration: float | None, given: list[tuple[str, str | None]] | None, make_video: bool,
           log=print) -> dict:
    """Everything after detection: teams, ball, possession, files."""
    if not samples:
        raise SystemExit("No frames could be read from the video.")
    out.mkdir(parents=True, exist_ok=True)
    teams = fit_teams(samples, given)
    for s in samples:
        for p in s.people:
            p.team = OTHER if p.kind in ("goalkeeper", "referee") else teams.assign(p.colour)
    log(f"Teams: {teams.names[0]} (shirts {teams.colours[0]}), {teams.names[1]} (shirts {teams.colours[1]})")
    if not given:
        hint = ",".join(f"Name={colour_name(c)}" for c in teams.colours)
        log(f'To use real names, run again with --teams "{hint}" (the detections are reused).')
    if teams.warning:
        log("Warning: " + teams.warning)
    ball, source = clean_ball(samples, info["width"], hz)
    config = Path(__file__).resolve().parent.parent / "models" / "config.json"
    proba, states = possession(samples, ball, config, hz)
    summary = summarise(samples, states, source, teams, hz, video, info, weights)
    write_csv(out / "possession.csv", samples, proba, states, ball, source)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        from .video_report import write_report
        write_report(out / "report.html", summary)
    except ImportError:
        pass
    if make_video:
        log("Drawing annotated.mp4 ...")
        render(video, out / "annotated.mp4", samples, states, ball, teams, hz, start, duration)
    a, b = summary["teams"]
    log(f"Possession: {a['name']} {a['possession']:.0%}, {b['name']} {b['possession']:.0%}. "
        f"Ball seen in {summary['ball_seen_share']:.0%} of frames. Results in {out}/")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m possession.video",
                                 description="Possession from a football match video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: out/<video name>)")
    ap.add_argument("--model", default="yolo26m.pt",
                    help="YOLO weights: a COCO model (downloaded automatically) or a football model")
    ap.add_argument("--imgsz", type=int, default=1280, help="detector input size; the ball needs at least 1280")
    ap.add_argument("--hz", type=float, default=5.0, help="frames analysed per second of video")
    ap.add_argument("--device", default=None, help="'cpu', or a GPU number such as 0 (default: GPU if there is one)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--start", type=float, default=0.0, help="start at this second")
    ap.add_argument("--duration", type=float, default=None, help="analyse only this many seconds")
    ap.add_argument("--teams", default=None,
                    help='each team with its shirt colour, e.g. "Zamalek=white,Al Ahly=red" (a colour word or #hex)')
    ap.add_argument("--no-video", action="store_true", help="skip annotated.mp4")
    args = ap.parse_args(argv)
    try:
        parse_teams(args.teams)
    except ValueError as e:
        ap.error(str(e))
    out = args.out or Path("out") / args.video.stem
    analyse(args.video, out, args.model, args.hz, args.imgsz, args.device, args.batch, args.start,
            args.duration, args.teams, not args.no_video)


if __name__ == "__main__":
    sys.exit(main())

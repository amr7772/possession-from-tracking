"""Build the self-contained replay page for game 2 (web/index.html).

    python -m possession.replay      # after the pipeline has written results/

Everything the page needs is packed into one HTML file: player and ball
positions at 5 Hz as base64 int16 (decimetres), the three label tracks,
the model's probabilities, detected touches, the event log and running scores.
"""
from __future__ import annotations

import base64
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Match, load_game
from .features import compute_features
from .labels import ON_BALL
from .metrics import evaluate_segments

TEMPLATE = Path(__file__).with_name("replay_template.html")
SENTINEL = -32768
STEP = 5
CURVE_EVERY = 25  # samples (5 s)

LINKS = {
    "github": "https://github.com/amr7772/possession-from-tracking",
    "model": "https://huggingface.co/amreleraqi09/possession-from-tracking",
    "data": "https://github.com/metrica-sports/sample-data",
}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _decimetres(xy: np.ndarray) -> np.ndarray:
    v = np.round(xy * 10)
    return np.where(np.isfinite(v), v, SENTINEL).astype("<i2")


def _digits(labels: np.ndarray) -> str:
    return (np.asarray(labels, dtype=np.uint8) + 48).tobytes().decode("ascii")


def _ranges(mask: np.ndarray) -> list[list[int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return [[int(s), int(e)] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def encode_players(m: Match, sel: np.ndarray) -> tuple[list[dict], str]:
    slots, chunks, offset = [], [], 0
    for team, P, ids in (("home", m.home, m.home_ids), ("away", m.away, m.away_ids)):
        Ps = P[sel]
        for k, pid in enumerate(ids):
            ok = np.isfinite(Ps[:, k, 0])
            if not ok.any():
                continue
            s, e = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1]) + 1
            seg = _decimetres(Ps[s:e, k, :]).ravel()
            slots.append({"team": team, "id": pid, "start": s, "end": e, "offset": offset})
            chunks.append(seg)
            offset += seg.size
    return slots, _b64(np.concatenate(chunks))


def running_scores(y, model, base, mask, period) -> dict:
    n = len(y)
    keys = ("macro_f1", "spell_f1", "spells_matched", "spells_true", "spells_pred")
    out = {f"{who}_{k}": [] for who in ("model", "base") for k in keys}
    for k in range(1, math.ceil(n / CURVE_EVERY) + 1):
        upto = min(k * CURVE_EVERY, n)
        mk = mask.copy()
        mk[upto:] = False
        for who, pred in (("model", model), ("base", base)):
            if mk.sum() < 10:
                vals = dict.fromkeys(keys, 0)
            else:
                vals = evaluate_segments(y, pred, mk, period)
            for key in keys:
                v = vals[key]
                out[f"{who}_{key}"].append(round(float(v), 3) if isinstance(v, float) else int(v))
    out["every"] = CURVE_EVERY
    return out


def build_payload(game: dict, preds: pd.DataFrame, summary: dict) -> dict:
    m: Match = game["match"]
    sel = np.arange(0, len(m), STEP)
    y = preds["truth"].to_numpy()
    model = preds["model"].to_numpy()
    base = preds["baseline"].to_numpy()
    mask = preds["in_eval"].to_numpy().astype(bool)
    period = preds["period"].to_numpy()
    assert len(sel) == len(preds)

    slots, players_b64 = encode_players(m, sel)
    ev = m.events
    periods = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        kick = int(ev[(ev.period == p) & ev.type.isin(ON_BALL)].start_frame.min())
        periods.append({"period": p, "start": int(idx[0]), "end": int(idx[-1]) + 1,
                        "kickoff": int((kick - m.frame[0]) // STEP)})

    touches = game["touches"]
    events = [[int((r.start_frame - m.frame[0]) // STEP), 0 if r.team == "Home" else 1,
               r.type, r.subtype, r.from_player.replace("Player", ""), r.to_player.replace("Player", "")]
              for r in ev.sort_values("start_frame", kind="stable").itertuples(index=False)]

    proba = preds[["p_home", "p_away", "p_dead"]].to_numpy()
    q = np.clip(np.round(proba * 255), 0, 255).astype(np.uint8)
    return {
        "n": int(len(sel)),
        "hz": 25 // STEP,
        "players": players_b64,
        "slots": slots,
        "ball": _b64(_decimetres(m.ball[sel])),
        "truth": _digits(y),
        "model": _digits(model),
        "baseline": _digits(base),
        "p_home": _b64(q[:, 0]),
        "p_away": _b64(q[:, 1]),
        "p_dead": _b64(q[:, 2]),
        "mask": _ranges(mask),
        "periods": periods,
        "touches": [[int(i // STEP), int(t)] for i, t in zip(touches["idx"], touches["team"])],
        "events": events,
        "curve": running_scores(y, model, base, mask, period),
        "summary": summary,
        "links": LINKS,
    }


def render(payload: dict, out_dir: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/{}", blob)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"  replay written to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def build_replay(game: dict, run: dict, summary: dict, out_dir: Path) -> Path:
    preds = pd.DataFrame({
        "period": game["period"], "in_eval": game["mask"].astype(int), "truth": game["y"],
        "model": run["pred"], "baseline": game["baseline"],
        # same rounding as results/predictions_game2.csv, so both build paths match byte for byte
        "p_home": run["proba"][:, 0].round(4), "p_away": run["proba"][:, 1].round(4),
        "p_dead": run["proba"][:, 2].round(4),
    })
    return render(build_payload(game, preds, summary), out_dir)


def main() -> None:
    root = Path(".")
    preds = pd.read_csv(root / "results" / "predictions_game2.csv")
    summary = json.loads((root / "results" / "metrics.json").read_text())
    m = load_game(2, root / "data" / "raw")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, touches = compute_features(m)
    game = {"match": m, "touches": touches}
    render(build_payload(game, preds, summary), root / "web")


if __name__ == "__main__":
    main()

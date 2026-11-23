"""End to end: download, label, build features, train on one match, test on the other.

    python -m possession.pipeline            # train on game 1, test on game 2 (+ the reverse)

Writes the model to models/, scores to results/ and the replay page to web/.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import nearest_player
from .data import download_game, load_game
from .features import compute_features, feature_columns
from .labels import CLASS_NAMES, possession_labels
from .metrics import evaluate_segments
from .model import out_of_fold_proba, save, train_classifier, tune_smoother

STEP = 5  # 25 Hz -> 5 Hz


def prepare(game: int, data_dir: Path) -> dict:
    t0 = time.time()
    m = load_game(game, data_dir)
    labels, mask = possession_labels(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats, touches = compute_features(m)
    sel = np.arange(0, len(m), STEP)
    print(f"  game {game}: {len(m):,} frames, {len(touches):,} touches detected ({time.time() - t0:.1f}s)")
    return {
        "game": game, "match": m, "sel": sel,
        "X": feats.iloc[sel][feature_columns(feats)].reset_index(drop=True),
        "y": labels[sel], "mask": mask[sel], "period": m.period[sel],
        "baseline": nearest_player(m)[sel], "touches": touches,
    }


def rounded(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def run(train: dict, test: dict) -> dict:
    print(f"  train on game {train['game']}, test on game {test['game']}")
    oof = out_of_fold_proba(train["X"], train["y"])
    smoother = tune_smoother(oof, train["y"], train["mask"], train["period"])
    print(f"    smoother: stickiness={smoother.stickiness}, alpha={smoother.alpha}")
    booster = train_classifier(train["X"], train["y"])
    proba = booster.predict(test["X"])
    pred = smoother.decode(proba, test["period"])
    ev = lambda p: rounded(evaluate_segments(test["y"], p, test["mask"], test["period"]))
    scores = {
        "model": ev(pred),
        "model_without_smoothing": ev(proba.argmax(axis=1)),
        "baseline_nearest_player": ev(test["baseline"]),
        "train_out_of_fold": rounded(evaluate_segments(
            train["y"], smoother.decode(oof, train["period"]), train["mask"], train["period"])),
    }
    for name in ("model", "model_without_smoothing", "baseline_nearest_player"):
        s = scores[name]
        print(f"    {name:26s} macro-F1 {s['macro_f1']:.3f}  spell-F1 {s['spell_f1']:.3f} "
              f"({s['spells_matched']}/{s['spells_true']} found, {s['spells_pred']} predicted)")
    return {"booster": booster, "smoother": smoother, "proba": proba, "pred": pred, "scores": scores}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args(argv)

    print("data")
    for g in (1, 2):
        download_game(g, args.data)
    games = {g: prepare(g, args.data) for g in (1, 2)}

    print("model")
    main_run = run(games[1], games[2])
    reverse = run(games[2], games[1])

    features = list(games[1]["X"].columns)
    save(main_run["booster"], main_run["smoother"], features, STEP, args.out / "models",
         extra={"trained_on": "Metrica Sports sample game 1", "tested_on": "Metrica Sports sample game 2"})

    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_game_1_test_game_2": main_run["scores"],
        "train_game_2_test_game_1": reverse["scores"],
        "smoother": {"stickiness": main_run["smoother"].stickiness, "alpha": main_run["smoother"].alpha},
        "classes": list(CLASS_NAMES),
        "sample_rate_hz": 25 // STEP,
    }
    (results / "metrics.json").write_text(json.dumps(summary, indent=2))
    (results / "smoother_tuning.json").write_text(json.dumps(main_run["smoother"].tuning, indent=2))

    gain = main_run["booster"].feature_importance("gain")
    imp = pd.DataFrame({"feature": features, "gain": gain / gain.sum()}).sort_values("gain", ascending=False)
    imp.to_csv(results / "feature_importance.csv", index=False, float_format="%.5f")

    g2 = games[2]
    preds = pd.DataFrame({
        "frame": g2["match"].frame[g2["sel"]], "period": g2["period"],
        "time_s": g2["match"].time[g2["sel"]].round(2), "in_eval": g2["mask"].astype(int),
        "truth": g2["y"], "model": main_run["pred"], "baseline": g2["baseline"],
        "p_home": main_run["proba"][:, 0].round(4), "p_away": main_run["proba"][:, 1].round(4),
        "p_dead": main_run["proba"][:, 2].round(4),
    })
    preds.to_csv(results / "predictions_game2.csv", index=False)

    if not args.no_replay:
        from .replay import build_replay
        build_replay(g2, main_run, summary, args.out / "web")
    print("done")


if __name__ == "__main__":
    main()

---
license: mit
library_name: lightgbm
pipeline_tag: tabular-classification
tags:
- football
- soccer
- sports-analytics
- tracking-data
- lightgbm
metrics:
- f1
---

# Who has the ball? Possession from tracking data

A LightGBM classifier with a small hidden Markov model on top. Five times a second it says whether the home team has the ball, the away team has it, or the ball is dead. Its only input is tracking data, the positions of the players and the ball.

- Code, training and evaluation: https://github.com/amr7772/possession-from-tracking
- Replay of the test match: https://huggingface.co/spaces/amreleraqi09/possession-from-tracking

## Files

- `possession_lgbm.txt`: the LightGBM booster in text format. Three classes, in this order: home, away, dead.
- `config.json`: the 78 feature names in the order the booster expects them, the sampling rate (every fifth frame of 25 Hz tracking) and the smoothing settings (class prior, transition matrix, alpha).
- `metrics.json`: scores on the held-out match, and on the reverse split.

## Results

Trained on Metrica Sports sample game 1, tested on game 2:

| | Spell F1 | Spells found | Spells predicted | Frame macro-F1 |
|---|---:|---:|---:|---:|
| This model | **0.817** | 257 of 347 | 282 | **0.956** |
| Same model, no smoothing | 0.761 | 271 of 347 | 365 | 0.957 |
| Nearest-player baseline | 0.100 | 114 of 347 | 1,933 | 0.849 |

A possession spell is an unbroken run of one team on the ball. A predicted spell counts as found when it overlaps a true spell of the same team with an intersection-over-union of 0.5 or more. Trained on game 2 and tested on game 1, the model scores 0.837 spell F1 against 0.135 for the baseline.

## Use

The features are built by `compute_features` in the GitHub repository, so clone it first.

```python
from huggingface_hub import snapshot_download
from possession.data import load_game
from possession.features import compute_features
from possession.model import load

snapshot_download("amreleraqi09/possession-from-tracking", local_dir="models")
booster, smoother, config = load("models")

match = load_game(2, "data/raw")          # any tracking data in the same Match structure
feats, _ = compute_features(match)
rows = feats.iloc[::config["sample_every_n_frames"]]
proba = booster.predict(rows[config["features"]])
labels = smoother.decode(proba, rows["period"].to_numpy())   # 0 home, 1 away, 2 dead
```

## Training data

One match: Metrica Sports sample game 1, anonymised, 25 Hz tracking. Labels come from the match's event log. The smoothing settings were tuned on out-of-fold predictions from the same match, and game 2 was kept back for testing.

## Limitations

- Trained on a single match from one provider. Expect it to need retraining for other tracking systems.
- Metrica stops tracking the ball when it goes out of play, and the model leans on that to spot dead balls. Broadcast tracking has no such signal.
- The labels inherit the event log's timing, which can be a few frames off.

Data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).

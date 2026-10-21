"""Where does the model's gain come from? Playlist recall@10 on the test releases, split into
debut artists (no earlier release in the catalogue) and artists with history, for the served
model and the strongest heuristic. Bootstrap 95% intervals on the difference.

    python scripts/cold_start.py --bundle bundles/current --out reports/cold_start.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from mci import data
from mci import evaluate as ev
from mci.features import Featurizer
from mci.serving.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="bundles/current")
    ap.add_argument("--out", default="reports/cold_start.json")
    a = ap.parse_args()
    cat = data.load(Path(os.environ.get("MCI_DATA", "data/spotify_songs.csv")))
    hist, test = cat.history(), cat.split("test")
    eng = Engine.load(Path(a.bundle), precision="int8")
    vec, _ = eng.run(eng.featurizer.tracks(test))
    truth = [cat.membership[t] for t in test["track_id"]]
    model = ev.per_track_playlist(vec @ eng.playlist_vectors.T, truth)["recall@10"]
    tau = json.loads(Path("reports/training.json").read_text())["baselines"]["recency_half_life_days"]
    f = Featurizer.fit(hist, [])
    w = np.exp(-hist["age_days"].to_numpy(float) / tau) if tau else None
    knn = ev.audio_knn(f.audio(hist), hist["track_id"].tolist(), f.audio(test), cat.membership, len(cat.playlists), weights=w)
    base = ev.per_track_playlist(ev.artist_then_audio(hist, test, cat.membership, knn), truth)["recall@10"]
    debut = test["artist_prior_tracks"].to_numpy() == 0
    out = {}
    for name, m in (("debut artists", debut), ("artists with history", ~debut), ("all", np.ones_like(debut))):
        d, lo, hi = ev.paired_bootstrap(model[m], base[m])
        out[name] = {"tracks": int(m.sum()), "model_recall@10": round(float(model[m].mean()), 4), "heuristic_recall@10": round(float(base[m].mean()), 4),
                     "gain": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Does learning from live listening pay off, and does the offline model help it start?

A semi-synthetic benchmark. The tracks, their real popularity, and which playlists really took
them are all from the catalogue's newest releases (the test period). Only the listeners are
simulated: in each playlist, a play of a track completes with probability

    p = sigmoid(-0.6 + 0.5 * popularity_z + 1.0 * [the playlist really placed this track] + noise)

where `noise` (sd 0.35) is a per-(playlist, track) effect nobody can predict before release.
Every hour, every playlist gets 300 plays. Each policy chooses what to play, and outcomes reach
the learner in batches of 25 plays, as they would through the event stream.

Policies:
  - random;
  - model only: always play what the offline model ranks highest, never learn;
  - learning only: Thompson sampling from a flat prior;
  - model + learning: Thompson sampling, prior from the offline model (what the service runs);
  - oracle: knows p (the ceiling, for regret).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mci import data
from mci.online import MemoryStats, ThompsonPolicy, prior_from_scores
from mci.serving.engine import Engine

PLAYS_PER_HOUR = 300
BATCH = 25
POOL = 60
N_CONTEXTS = 20


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def world(cat: data.Catalogue, engine: Engine, seed: int):
    rng = np.random.default_rng(seed)
    test = cat.split("test").reset_index(drop=True)
    counts: dict[int, list[int]] = {}
    for i, t in enumerate(test["track_id"]):
        for p in cat.membership[t]:
            counts.setdefault(p, []).append(i)
    contexts = sorted((p for p in counts if len(counts[p]) >= 15), key=lambda p: -len(counts[p]))[:N_CONTEXTS]
    vec, pop = engine.run(engine.featurizer.tracks(test))
    pop_z = _z(test["track_popularity"].to_numpy(float))
    out = []
    for p in contexts:
        pos = counts[p][: POOL // 2]
        others = rng.choice(np.setdiff1d(np.arange(len(test)), counts[p]), POOL - len(pos), replace=False)
        pool = np.array([*pos, *others])
        member = np.isin(pool, counts[p]).astype(float)
        true_p = 1 / (1 + np.exp(-(-0.6 + 0.5 * pop_z[pool] + 1.0 * member + rng.normal(0, 0.35, len(pool)))))
        j = engine.playlist_index[cat.playlists.loc[p, "playlist_id"]]
        model_score = _z(vec[pool] @ engine.playlist_vectors[j]) + _z(pop[pool])
        out.append({"context": str(p), "tracks": [str(i) for i in pool], "p": true_p, "member": member, "prior": prior_from_scores(model_score)})
    return out


def simulate(worlds, policy: str, hours: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1000)
    stats = MemoryStats()
    tp = ThompsonPolicy(stats, rng=rng)
    flat = None
    plays = completions = fit_plays = 0
    first6 = [0, 0]
    hourly = np.zeros(hours)
    for h in range(hours):
        for w in worlds:
            if flat is None or len(flat) != len(w["tracks"]):
                flat = np.full(len(w["tracks"]), 0.5)
            for _ in range(PLAYS_PER_HOUR // BATCH):
                if policy == "random":
                    pick = rng.integers(0, len(w["tracks"]), BATCH)
                elif policy == "model only":
                    pick = np.full(BATCH, int(np.argmax(w["prior"])))
                elif policy == "oracle":
                    pick = np.full(BATCH, int(np.argmax(w["p"])))
                elif policy == "learning only":
                    tp.strength = 2.0  # Beta(1, 1)
                    pick = tp.choose_each(w["context"], w["tracks"], flat, BATCH)
                else:
                    tp.strength = 20.0
                    pick = tp.choose_each(w["context"], w["tracks"], w["prior"], BATCH)
                done = rng.random(BATCH) < w["p"][pick]
                for i in np.unique(pick):
                    sel = pick == i
                    stats.add(w["context"], w["tracks"][i], int(sel.sum()), int(done[sel].sum()))
                plays += BATCH
                completions += int(done.sum())
                fit_plays += int(w["member"][pick].sum())
                hourly[h] += done.sum()
                if h < 6:
                    first6[0] += BATCH
                    first6[1] += int(done.sum())
    return {"completion_rate": completions / plays, "first_6h_completion_rate": first6[1] / first6[0], "share_of_plays_to_real_fits": fit_plays / plays,
            "hourly_completions": hourly.tolist(), "completions": completions}  # fmt: skip


def run(csv: Path, bundle: Path | None = None, hours: int = 72, seeds: int = 5) -> dict:
    import os

    cat = data.load(csv)
    engine = Engine.load(bundle or Path(os.environ.get("MCI_BUNDLE", "bundles/current")), precision="int8")
    policies = ["random", "model only", "learning only", "model + learning", "oracle"]
    t0 = time.time()
    res: dict[str, list[dict]] = {p: [] for p in policies}
    for s in range(seeds):
        w = world(cat, engine, s)
        for p in policies:
            res[p].append(simulate(w, p, hours, s))
    summary = {}
    for p in policies:
        r = res[p]

        def agg(k, rows=r):
            v = np.array([x[k] for x in rows])
            return {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

        summary[p] = {k: agg(k) for k in ("completion_rate", "first_6h_completion_rate", "share_of_plays_to_real_fits")}
        summary[p]["regret_vs_oracle"] = round(float(np.mean([o["completions"] - x["completions"] for o, x in zip(res["oracle"], r, strict=True)])), 1)
    curves = {p: np.mean([x["hourly_completions"] for x in res[p]], axis=0).round(1).tolist() for p in policies}
    return {"setup": {"contexts": N_CONTEXTS, "candidates_per_context": POOL, "plays_per_hour_per_context": PLAYS_PER_HOUR, "learner_batch": BATCH,
                      "hours": hours, "seeds": seeds, "model": engine.version, "seconds": round(time.time() - t0, 1)},
            "summary": summary, "hourly_completions_mean": curves}  # fmt: skip

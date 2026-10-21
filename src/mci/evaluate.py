"""Metrics, baselines and the promotion gate.

Two questions, both asked of tracks released after everything the models trained on:
  - playlist matching: rank all 471 playlists for a new track; did its real playlists come out on
    top? (recall@10, MRR)
  - performance: how popular will it be? (MAE, Spearman rank correlation, and AUC for "hit",
    a top-20% track)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

K = 10


# ------------------------------------------------------------------------------------- metrics
def per_track_playlist(scores: np.ndarray, truth: list[list[int]], k: int = K) -> dict[str, np.ndarray]:
    """scores [n, playlists]; truth = true playlist indices per track. Per-track values, so the
    gate can bootstrap differences between models on the same tracks."""
    order = np.argsort(-scores, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])
    recall = np.array([np.mean([rank[i, p] < k for p in ps]) for i, ps in enumerate(truth)])
    rr = np.array([1.0 / (1 + min(rank[i, p] for p in ps)) for i, ps in enumerate(truth)])
    return {f"recall@{k}": recall, "mrr": rr}


def per_track_popularity(pred: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {"abs_error": np.abs(pred - y)}


def summarise(pl: dict[str, np.ndarray] | None, pop_pred: np.ndarray | None, df: pd.DataFrame) -> dict:
    out = {}
    if pl is not None:
        out.update({k: round(float(v.mean()), 4) for k, v in pl.items()})
    if pop_pred is not None:
        y = df["track_popularity"].to_numpy(float)
        out["mae"] = round(float(np.abs(pop_pred - y).mean()), 3)
        out["spearman"] = round(float(stats.spearmanr(pop_pred, y).statistic), 4)
        out["hit_auc"] = round(float(roc_auc_score(df["hit"], pop_pred)), 4)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) and its 95% interval, resampling tracks."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ------------------------------------------------------------------------------------- baselines
def most_popular_playlists(train: pd.DataFrame, membership, n_playlists: int, n: int) -> np.ndarray:
    counts = np.zeros(n_playlists)
    for t in train["track_id"]:
        counts[membership[t]] += 1
    return np.tile(counts, (n, 1))


def recent_playlists(history: pd.DataFrame, membership, n_playlists: int, n: int, days: int = 90) -> np.ndarray:
    """Playlists that took the most releases in the last `days` of history: where new music goes."""
    last = history["release_date"].max()
    recent = history[history["release_date"] > last - pd.Timedelta(days=days)]
    return most_popular_playlists(recent, membership, n_playlists, n)


def audio_knn(train_x: np.ndarray, train_ids, query_x: np.ndarray, membership, n_playlists: int, k: int = 50, weights: np.ndarray | None = None) -> np.ndarray:
    """Playlists of the k most similar training tracks by audio descriptors (cosine), weighted by
    similarity: what an A&R person does with a reference track."""
    a = train_x / (np.linalg.norm(train_x, axis=1, keepdims=True) + 1e-9)
    q = query_x / (np.linalg.norm(query_x, axis=1, keepdims=True) + 1e-9)
    member = np.zeros((len(train_ids), n_playlists), np.float32)
    for i, t in enumerate(train_ids):
        member[i, membership[t]] = 1.0
    out = np.zeros((len(q), n_playlists), np.float32)
    for s in range(0, len(q), 512):
        sim = q[s : s + 512] @ a.T
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        w = np.take_along_axis(sim, top, 1).clip(0)
        if weights is not None:
            w = w * weights[top]
        out[s : s + 512] = np.einsum("nk,nkp->np", w, member[top])
    return out


def artist_then_audio(train: pd.DataFrame, query: pd.DataFrame, membership, knn_scores: np.ndarray) -> np.ndarray:
    """The strongest heuristic: playlists the artist was placed in before (by how often), then the
    audio neighbours (recency-weighted) for everything else and for debut artists."""
    history: dict[str, np.ndarray] = {}
    n_playlists = knn_scores.shape[1]
    for artist, g in train.groupby("track_artist"):
        c = np.zeros(n_playlists)
        for t in g["track_id"]:
            c[membership[t]] += 1
        history[artist] = c
    scale = knn_scores.max(1, keepdims=True) + 1e-9
    out = knn_scores / scale
    for i, a in enumerate(query["track_artist"]):
        if a in history:
            out[i] = out[i] + 10.0 * history[a] / history[a].max()
    return out


def popularity_baselines(train: pd.DataFrame, query: pd.DataFrame) -> dict[str, np.ndarray]:
    mean = train["track_popularity"].mean()
    prior = query["artist_prior_pop"].to_numpy(float)
    n = query["artist_prior_tracks"].to_numpy(float)
    shrink = n / (n + 3.0)  # trust an artist's history more the longer it is
    return {"training mean": np.full(len(query), mean), "artist history (shrunk)": np.where(n > 0, shrink * np.nan_to_num(prior) + (1 - shrink) * mean, mean)}


def lightgbm_popularity(train_x: np.ndarray, train_y: np.ndarray, valid_x, valid_y, query_x: np.ndarray, seed: int = 0) -> np.ndarray:
    import lightgbm as lgb

    m = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    m.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(query_x)


# ------------------------------------------------------------------------------------- gate
GATE = {"min_recall_gain": 0.0, "min_mae_gain": 0.0, "max_p99_ms": 50.0}


def gate(candidate: dict, champion: dict) -> dict:
    """Promote only when the candidate is better on BOTH tasks with 95% confidence (paired
    bootstrap over test tracks) and serves fast enough. `candidate`/`champion` hold per-track
    arrays: recall, abs_error; and the candidate's measured latency."""
    r = paired_bootstrap(candidate["recall"], champion["recall"])
    e = paired_bootstrap(champion["abs_error"], candidate["abs_error"])  # positive = candidate better
    checks = {
        "playlist_recall_better": {"ok": r[1] > GATE["min_recall_gain"], "gain": round(r[0], 4), "ci95": [round(r[1], 4), round(r[2], 4)]},
        "popularity_error_lower": {"ok": e[1] > GATE["min_mae_gain"], "gain": round(e[0], 3), "ci95": [round(e[1], 3), round(e[2], 3)]},
        "fast_enough": {"ok": candidate.get("p99_ms", 0.0) <= GATE["max_p99_ms"], "p99_ms": candidate.get("p99_ms")},
    }
    return {"promote": all(c["ok"] for c in checks.values()), "checks": checks, "thresholds": GATE}

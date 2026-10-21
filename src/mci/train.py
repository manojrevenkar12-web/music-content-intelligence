"""Training pipeline: select on the validation quarter, refit on all history, evaluate on the newest
releases, export ONNX (fp32 and int8), write the serving bundle, record everything in MLflow.

    mci train                        # full pipeline: candidates, selection, bundle, gate, registry
    mci train --quick                # CI smoke run: small, fast, same code path

Each task gets its own champion, chosen on the validation quarter (never on the test releases):
  - playlist matching: the multimodal two-tower variants (also the ablation study);
  - popularity: LightGBM on audio + context, the neural model selected for popularity, or the mean
    of the two.
The test releases are scored once, by everything, for the report and the gate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mci import data
from mci import evaluate as ev
from mci.features import Featurizer
from mci.model import TrackExport, TrackModel

log = logging.getLogger("mci.train")


@dataclass
class Config:
    use_text: bool = True
    use_pretrained: bool = True
    finetune: bool = True
    epochs: int = 30
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    pop_weight: float = 1.0  # auxiliary for the playlist model; the popularity model raises it
    drop_ids: float = 0.3
    patience: int = 5
    seed: int = 7
    select_on: str = "playlist"  # early stopping: "playlist" (recall@10) or "popularity" (MAE)


VARIANTS = {
    "audio + context only": Config(use_text=False),
    "text, no pretrained vectors": Config(use_pretrained=False),
    "text, pretrained frozen": Config(finetune=False),
    "text, pretrained fine-tuned": Config(),
}
NEURAL_POPULARITY = Config(pop_weight=8.0, select_on="popularity")


# ------------------------------------------------------------------------------------- inputs
def pretrained_matrix(vocab: dict[str, int], dim: int = 300) -> np.ndarray:
    """Rows of spaCy's en_core_web_md vectors for the vocabulary (row 0 padding, 1 unknown)."""
    m = np.zeros((len(vocab) + 2, dim), np.float32)
    try:
        import spacy

        nlp = spacy.load("en_core_web_md", exclude=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler", "senter", "tok2vec"])
    except (ImportError, OSError):
        log.warning("en_core_web_md not installed: training without pretrained vectors")
        return m
    for w, i in vocab.items():
        lex = nlp.vocab[w]
        if lex.has_vector:
            m[i] = lex.vector
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)  # unit length: the projection learns the scale
    return m


def has_vector_fn():
    try:
        import spacy

        nlp = spacy.load("en_core_web_md", exclude=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler", "senter", "tok2vec"])
        return lambda w: nlp.vocab.has_vector(w)
    except (ImportError, OSError):
        return lambda w: False


def tensors(f: Featurizer, df: pd.DataFrame) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in f.tracks(df).items()}


def targets(df: pd.DataFrame, membership, n_playlists: int) -> torch.Tensor:
    y = np.zeros((len(df), n_playlists), np.float32)
    for i, t in enumerate(df["track_id"]):
        y[i, membership[t]] = 1.0
    return torch.from_numpy(y / y.sum(1, keepdims=True))


# ------------------------------------------------------------------------------------- fitting
@dataclass
class Fitted:
    model: TrackModel
    featurizer: Featurizer
    playlist_inputs: tuple[torch.Tensor, torch.Tensor]
    epochs_run: int
    history: list[dict]
    valid_best: dict | None = None


def _predict(m: TrackModel, x: dict[str, torch.Tensor], pl: tuple[torch.Tensor, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    with torch.no_grad():
        pv = m.playlist_vectors(*pl)
        scores, pops = [], []
        n = x["audio"].shape[0]
        for s in range(0, n, 2048):
            b = {k: v[s : s + 2048] for k, v in x.items()}
            h = m.track_hidden(b)
            scores.append((m.track_vector(h) @ pv.T).numpy())
            pops.append(m.popularity(h, b["context"]).numpy())
    return np.concatenate(scores), np.concatenate(pops)


def fit(cfg: Config, cat: data.Catalogue, train: pd.DataFrame, valid: pd.DataFrame | None, epochs: int | None = None, has_vector=None) -> Fitted:
    """Train on `train`. With `valid`, stop early on validation; without, run exactly `epochs`."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    names = cat.playlists["playlist_name"].tolist()
    f = Featurizer.fit(train, names, has_vector or (lambda w: False))
    pre = pretrained_matrix(f.vocab) if cfg.use_pretrained else np.zeros((len(f.vocab) + 2, 300), np.float32)
    n_pl = len(cat.playlists)
    m = TrackModel(pre, n_pl, cfg.use_text, cfg.use_pretrained, cfg.finetune)
    x, y = tensors(f, train), targets(train, cat.membership, n_pl)
    pop = torch.tensor(train["track_popularity"].to_numpy(np.float32))
    pl = tuple(torch.from_numpy(a) for a in f.text(names, "playlist"))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = epochs or cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total * ((len(train) + cfg.batch - 1) // cfg.batch))
    gen = torch.Generator().manual_seed(cfg.seed)
    vx = tensors(f, valid) if valid is not None else None
    best, best_state, best_epoch, hist, bad, best_summ = -1e9, None, total, [], 0, None
    for epoch in range(1, total + 1):
        m.train()
        t0, losses = time.time(), []
        for idx in torch.randperm(len(train), generator=gen).split(cfg.batch):
            b = {k: v[idx] for k, v in x.items()}
            h = m.track_hidden(b)
            logits = m.scale() * m.track_vector(h) @ m.playlist_vectors(*pl, drop_ids=cfg.drop_ids).T
            l_pl = -(y[idx] * torch.log_softmax(logits, 1)).sum(1).mean()
            l_pop = torch.nn.functional.smooth_l1_loss(m.popularity(h, b["context"]) / 100.0, pop[idx] / 100.0, beta=0.1)
            loss = l_pl + cfg.pop_weight * l_pop
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if vx is not None:
            s, p = _predict(m, vx, pl)
            summ = ev.summarise(ev.per_track_playlist(s, [cat.membership[t] for t in valid["track_id"]]), p, valid)
            row.update(summ)
            score = summ["recall@10"] if cfg.select_on == "playlist" else -summ["mae"]
            if score > best:
                best, best_epoch, bad, best_summ = score, epoch, 0, summ
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
        hist.append(row)
        log.info("epoch %s", row)
        if vx is not None and bad >= cfg.patience:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return Fitted(m, f, pl, best_epoch, hist, best_summ)


# ------------------------------------------------------------------------------------- export
def export_lightgbm(model, path: Path) -> None:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    from mci.features import N_AUDIO, N_CONTEXT

    onx = onnxmltools.convert_lightgbm(model, initial_types=[("tabular", FloatTensorType([None, N_AUDIO + N_CONTEXT]))], target_opset=15)
    path.write_bytes(onx.SerializeToString())


def export_onnx(fitted: Fitted, path: Path) -> None:
    exp = TrackExport(fitted.model).eval()
    sample = {k: v[:4] for k, v in tensors(fitted.featurizer, _example_frame()).items()}
    names = ["audio", "context", "title_words", "title_tris", "artist_words", "artist_tris"]
    torch.onnx.export(
        exp, tuple(sample[n] for n in names), str(path), input_names=names, output_names=["track_vector", "popularity"],
        dynamic_axes={**{n: {0: "batch"} for n in names}, "track_vector": {0: "batch"}, "popularity": {0: "batch"}}, opset_version=17, dynamo=False,
    )  # fmt: skip


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "Gather"])


def _example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_name": ["Night Drive", "Corazón", "Intro", "Another Song (feat. Someone) - Remix"],
            "track_album_name": ["Night Drive", "Corazón", "", "Another Song"],
            "track_artist": ["Example", "Ejemplo", "X", "Somebody"],
            "danceability": [0.6] * 4, "energy": [0.7] * 4, "loudness": [-6.0] * 4, "speechiness": [0.05] * 4, "acousticness": [0.1] * 4,
            "instrumentalness": [0.0] * 4, "liveness": [0.1] * 4, "valence": [0.5] * 4, "tempo": [120.0] * 4, "duration_ms": [200_000] * 4,
            "mode": [1] * 4, "key": [5] * 4, "artist_prior_tracks": [0, 2, 0, 5], "artist_prior_pop": [np.nan, 50.0, np.nan, 40.0], "age_days": [0] * 4,
        }
    )  # fmt: skip


def write_bundle(fitted: Fitted, cat: data.Catalogue, history: pd.DataFrame, out: Path, version: str, metrics: dict, extra: dict,
                 popularity: str = "neural", lgbm=None, pop_fitted: Fitted | None = None) -> Path:  # fmt: skip
    """Everything serving needs, and nothing it doesn't: ONNX graphs, featuriser, playlist vectors,
    the artist-history table for point-in-time context, and a manifest that says where it came from.

    `popularity` says which model answers "how popular": "neural" (the track model's own head),
    "lightgbm", "neural-pop" (the popularity-selected network) or an "ensemble" of the last two."""
    out.mkdir(parents=True, exist_ok=True)
    export_onnx(fitted, out / "model.onnx")
    quantize(out / "model.onnx", out / "model.int8.onnx")
    if lgbm is not None:
        export_lightgbm(lgbm, out / "popularity_lgbm.onnx")
    if pop_fitted is not None:
        export_onnx(pop_fitted, out / "popularity_nn.onnx")
        quantize(out / "popularity_nn.onnx", out / "popularity_nn.int8.onnx")
    fitted.featurizer.save(out / "featurizer.json")
    fitted.model.eval()
    with torch.no_grad():
        pv = fitted.model.playlist_vectors(*fitted.playlist_inputs).numpy().astype(np.float32)
    np.save(out / "playlist_vectors.npy", pv)
    pls = cat.playlists[["idx", "playlist_id", "playlist_name", "playlist_genre", "playlist_subgenre"]]
    (out / "playlists.json").write_text(json.dumps(pls.to_dict("records"), ensure_ascii=False))
    g = history.groupby("track_artist")["track_popularity"].agg(["count", "mean"]).reset_index()
    artists = {a: [int(c), round(float(m), 2)] for a, c, m in g.itertuples(index=False)}
    (out / "artists.json").write_text(json.dumps(artists, ensure_ascii=False))
    from mci import drift

    (out / "drift_reference.json").write_text(json.dumps(drift.reference(fitted.featurizer.audio(history))))
    manifest = {
        "version": version, "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "dataset_sha256": data.SHA256,
        "history_until": str(history["release_date"].max().date()), "snapshot": cat.tracks.attrs.get("snapshot"),
        "n_playlists": int(len(pls)), "vector_dim": int(pv.shape[1]), "scale": float(fitted.model.scale()), "metrics": metrics, "popularity_model": popularity, **extra,
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out


# ------------------------------------------------------------------------------------- pipeline
def baselines(cat: data.Catalogue, history: pd.DataFrame, query: pd.DataFrame, f: Featurizer, tau: float) -> dict:
    n_pl = len(cat.playlists)
    truth = [cat.membership[t] for t in query["track_id"]]
    out: dict[str, dict] = {}
    rec = ev.recent_playlists(history, cat.membership, n_pl, len(query), days=30)
    out["recently active playlists"] = ev.per_track_playlist(rec, truth)
    w = np.exp(-history["age_days"].to_numpy(float) / tau) if tau else None
    knn = ev.audio_knn(f.audio(history), history["track_id"].tolist(), f.audio(query), cat.membership, n_pl, weights=w)
    out["audio neighbours (recency-weighted)"] = ev.per_track_playlist(knn, truth)
    out["artist history + audio neighbours"] = ev.per_track_playlist(ev.artist_then_audio(history, query, cat.membership, knn), truth)
    return out


def _tabular(f: Featurizer, d: pd.DataFrame) -> np.ndarray:
    return np.hstack([f.audio(d), f.context(d)]).astype(np.float32)


def lightgbm_fit(f: Featurizer, train: pd.DataFrame, valid: pd.DataFrame | None, n_estimators: int | None = None, seed: int = 0):
    import lightgbm as lgb

    params = dict(learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    if valid is not None:
        m = lgb.LGBMRegressor(n_estimators=3000, **params)
        m.fit(
            _tabular(f, train),
            train["track_popularity"],
            eval_set=[(_tabular(f, valid), valid["track_popularity"])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return m
    return lgb.LGBMRegressor(n_estimators=n_estimators, **params).fit(_tabular(f, train), train["track_popularity"])


def _mlflow_name(k: str) -> str:
    return k.replace("@", "_at_")


def _fit_variant(name: str, cfg: Config, cat, train, valid, history, test, truth, hv, quick) -> dict:
    """Select epochs on validation, refit on all history with that many epochs, score the tests."""
    import mlflow

    if quick:
        cfg = Config(**{**asdict(cfg), "epochs": 3})
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**asdict(cfg), "variant": name, "quick": quick})
        t0 = time.time()
        sel = fit(cfg, cat, train, valid, has_vector=hv)
        _, valid_pop = _predict(sel.model, tensors(sel.featurizer, valid), sel.playlist_inputs)  # selection fit: trained on train only
        for row in sel.history:
            mlflow.log_metrics({_mlflow_name(f"valid_{k}"): v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
        final = fit(cfg, cat, history, None, epochs=sel.epochs_run, has_vector=hv)
        s, p = _predict(final.model, tensors(final.featurizer, test), final.playlist_inputs)
        per = ev.per_track_playlist(s, truth)
        summ = {**ev.summarise(per, p, test), "epochs": sel.epochs_run, "train_seconds": round(time.time() - t0, 1)}
        mlflow.log_metrics({_mlflow_name(f"test_{k}"): v for k, v in summ.items() if isinstance(v, int | float)})
        log.info("%s: valid %s test %s", name, sel.valid_best, summ)
        return {"final": final, "per": per, "pop": p, "valid_pop": valid_pop, "valid": sel.valid_best, "test": summ, "run_id": mlflow.active_run().info.run_id}


def run(csv: Path, out_dir: Path, quick: bool = False, variants: list[str] | None = None) -> dict:
    import mlflow

    cat = data.load(csv)
    train, valid, test, history = cat.split("train"), cat.split("valid"), cat.split("test"), cat.history()
    if quick:  # same pipeline on a slice: CI checks the code path, not the numbers
        train, valid, test, history = train.tail(3000), valid.head(600), test.head(600), pd.concat([train.tail(3000), valid.head(600)])
    hv = has_vector_fn()
    truth = [cat.membership[t] for t in test["track_id"]]
    y = test["track_popularity"].to_numpy(float)
    mlflow.set_experiment("music-content-intelligence")
    report: dict = {"dataset": {"tracks": len(cat.tracks), "playlists": len(cat.playlists), "train": len(train), "valid": len(valid), "test": len(test),
                                "history_until": str(history["release_date"].max().date()), "test_from": str(data.TEST_FROM.date()), "hit_threshold": cat.tracks.attrs["hit_threshold"]}}  # fmt: skip

    # 1. heuristic baselines: what a label does without ML. Their one knob is tuned on validation.
    names = cat.playlists["playlist_name"].tolist()
    f_train, f_hist = Featurizer.fit(train, names), Featurizer.fit(history, names)
    vtruth = [cat.membership[t] for t in valid["track_id"]]
    taus = {}
    for tau in (0, 30, 90, 180, 365):
        w = np.exp(-train["age_days"].to_numpy(float) / tau) if tau else None
        sc = ev.audio_knn(f_train.audio(train), train["track_id"].tolist(), f_train.audio(valid), cat.membership, len(cat.playlists), weights=w)
        taus[tau] = float(ev.per_track_playlist(ev.artist_then_audio(train, valid, cat.membership, sc), vtruth)["recall@10"].mean())
    tau = max(taus, key=taus.get)
    base_pl = baselines(cat, history, test, f_hist, tau)
    base_pop = ev.popularity_baselines(history, test)
    report["baselines"] = {"recency_half_life_days": tau, "playlist": {k: {m: round(float(a.mean()), 4) for m, a in v.items()} for k, v in base_pl.items()},
                           "popularity": {k: ev.summarise(None, p, test) for k, p in base_pop.items()}}  # fmt: skip

    # 2. playlist candidates (and the ablation study): the two-tower variants
    runs = {name: _fit_variant(name, VARIANTS[name], cat, train, valid, history, test, truth, hv, quick) for name in variants or list(VARIANTS)}
    report["playlist_candidates"] = {n: {"valid": r["valid"], "test": r["test"]} for n, r in runs.items()}
    pl_name = max(runs, key=lambda n: runs[n]["valid"]["recall@10"])

    # 3. popularity candidates: LightGBM, the popularity-selected network, and their mean
    lg_sel = lightgbm_fit(f_train, train, valid)
    lg_final = lightgbm_fit(f_hist, history, None, n_estimators=max(50, lg_sel.best_iteration_ or 50))
    nn = _fit_variant("popularity network", NEURAL_POPULARITY, cat, train, valid, history, test, truth, hv, quick)
    pop_valid = {"lightgbm": lg_sel.predict(_tabular(f_train, valid)), "neural-pop": nn["valid_pop"]}
    pop_valid["ensemble"] = (pop_valid["lightgbm"] + pop_valid["neural-pop"]) / 2
    pop_test = {"lightgbm": lg_final.predict(_tabular(f_hist, test)), "neural-pop": nn["pop"]}
    pop_test["ensemble"] = (pop_test["lightgbm"] + pop_test["neural-pop"]) / 2
    report["popularity_candidates"] = {k: {"valid": ev.summarise(None, pop_valid[k], valid), "test": ev.summarise(None, pop_test[k], test)} for k in pop_test}
    pop_name = min(pop_valid, key=lambda k: np.abs(pop_valid[k] - valid["track_popularity"].to_numpy(float)).mean())
    report["selected"] = {"playlist": pl_name, "popularity": pop_name, "selected_on": "validation quarter (2019-07-01 to 2019-09-30)"}

    # 4. bundle, gate against the best heuristic on each task, registry
    champ = runs[pl_name]
    version = time.strftime("mci-%Y%m%d-%H%M") if not quick else "mci-quick"
    bundle = write_bundle(champ["final"], cat, history, out_dir / version, version, {"playlist": champ["test"], "popularity": report["popularity_candidates"][pop_name]["test"]},
                          {"mlflow_run_id": champ["run_id"], "variant": pl_name}, popularity=pop_name,
                          lgbm=lg_final if pop_name in ("lightgbm", "ensemble") else None,
                          pop_fitted=nn["final"] if pop_name in ("neural-pop", "ensemble") else None)  # fmt: skip
    p99 = _latency_p99(bundle)
    best_pl = max(base_pl, key=lambda k: base_pl[k]["recall@10"].mean())
    best_pop = min(base_pop, key=lambda k: np.abs(base_pop[k] - y).mean())
    g = ev.gate(
        {"recall": champ["per"]["recall@10"], "abs_error": np.abs(pop_test[pop_name] - y), "p99_ms": p99},
        {"recall": base_pl[best_pl]["recall@10"], "abs_error": np.abs(base_pop[best_pop] - y)},
    )
    g["against"] = {"playlist": best_pl, "popularity": best_pop}
    report["gate"] = g
    report["int8"] = _int8_check(bundle, champ["final"], test, truth)
    report["bundle"] = {"version": version, "p99_ms_single_request_int8": p99, "bytes": {p.name: p.stat().st_size for p in sorted(bundle.iterdir())}}
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest.update({"gate": g, "int8": report["int8"]})
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    with mlflow.start_run(run_id=champ["run_id"]):
        import onnx

        info = mlflow.onnx.log_model(onnx.load(str(bundle / "model.int8.onnx")), name="track_model", registered_model_name="mci-track-model")
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
        mlflow.log_dict(g, "gate.json")
        if g["promote"]:
            mlflow.MlflowClient().set_registered_model_alias("mci-track-model", "champion", info.registered_model_version)
    report["registry"] = {"model": "mci-track-model", "version": info.registered_model_version, "alias": "champion" if g["promote"] else None}
    return report


def _latency_p99(bundle: Path) -> float:
    from mci.serving.engine import Engine

    eng = Engine.load(bundle, precision="int8")
    item = {"title": "Night Drive", "artist": "Example", "danceability": 0.6, "energy": 0.7, "loudness": -6, "speechiness": 0.05, "acousticness": 0.1,
            "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.5, "tempo": 120, "duration_ms": 200000, "mode": 1, "key": 5}  # fmt: skip
    for _ in range(20):
        eng.score([item])
    ts = []
    for _ in range(300):
        t0 = time.perf_counter()
        eng.score([item])
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.percentile(ts, 99)), 2)


def _int8_check(bundle: Path, final: Fitted, test: pd.DataFrame, truth: list[list[int]]) -> dict:
    """int8 must agree with fp32 closely enough to serve: same top-10 playlists, small popularity drift."""
    from mci.serving.engine import Engine

    a, b = Engine.load(bundle, precision="fp32"), Engine.load(bundle, precision="int8")
    x = final.featurizer.tracks(test)
    va, pa = a.run(x)
    vb, pb = b.run(x)
    ta = np.argsort(-(va @ a.playlist_vectors.T), 1)[:, :10]
    tb = np.argsort(-(vb @ b.playlist_vectors.T), 1)[:, :10]
    overlap = float(np.mean([len(set(r) & set(q)) / 10 for r, q in zip(ta, tb, strict=True)]))
    y = test["track_popularity"].to_numpy(float)
    rec = {p: float(ev.per_track_playlist(v @ e.playlist_vectors.T, truth)["recall@10"].mean()) for p, v, e in (("fp32", va, a), ("int8", vb, b))}
    return {"top10_overlap": round(overlap, 4), "recall@10_fp32": round(rec["fp32"], 4), "recall@10_int8": round(rec["int8"], 4),
            "popularity_mean_abs_diff": round(float(np.abs(pa - pb).mean()), 3),
            "mae_fp32": round(float(np.abs(pa - y).mean()), 3), "mae_int8": round(float(np.abs(pb - y).mean()), 3),
            "bytes_fp32": (bundle / "model.onnx").stat().st_size, "bytes_int8": (bundle / "model.int8.onnx").stat().st_size}  # fmt: skip

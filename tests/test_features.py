import numpy as np
import pytest

from mci import drift
from mci import evaluate as ev
from mci.features import MAX_TRIS, MAX_WORDS, Featurizer, request_frame, tokens, trigrams

from .conftest import api_track, track_rows


def test_tokens_keep_accents_and_drop_punctuation():
    assert tokens("Corazón (feat. J Balvin) - Remix") == ["corazón", "feat", "j", "balvin", "remix"]


def test_trigrams_are_stable_across_processes():
    # crc32, not Python's salted hash(): serving must hash exactly as training did
    assert trigrams(["amor"]) == trigrams(["amor"])
    assert trigrams(["amor"])[:2] == [1 + __import__("zlib").crc32(b"<am") % 32767, 1 + __import__("zlib").crc32(b"amo") % 32767]


def test_serving_path_matches_training_path():
    df = track_rows(30)
    f = Featurizer.fit(df, ["chill", "party"])
    items = [api_track(title=r.track_name, album=r.track_album_name, artist=r.track_artist, artist_prior_tracks=int(r.artist_prior_tracks),
                       artist_prior_pop=float(r.artist_prior_pop), age_days=int(r.age_days), **{c: getattr(r, c) for c in
                       ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "mode", "key"]})
             for r in df.itertuples()]  # fmt: skip
    a, b = f.tracks(df), f.tracks(request_frame(items))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-6, err_msg=k)
    assert a["title_words"].shape == (30, MAX_WORDS["title"]) and a["title_tris"].shape == (30, MAX_TRIS["title"])


def test_debut_artist_gets_neutral_context():
    f = Featurizer.fit(track_rows(30), [])
    x = f.tracks(request_frame([api_track(artist="Nobody Yet")]))
    np.testing.assert_allclose(x["context"][0], [0, 0, 0, 0])


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(5000, len(drift.FEATURES)))
    ref = drift.reference(train)
    same = drift.psi(ref["energy"], rng.normal(size=2000))
    shifted = drift.psi(ref["energy"], rng.normal(1.0, 1.0, size=2000))
    assert same < 0.02 and shifted > 0.25


def test_playlist_metrics():
    scores = np.array([[0.9, 0.1, 0.5], [0.1, 0.2, 0.3]])
    out = ev.per_track_playlist(scores, [[0], [0, 2]], k=1)
    assert out["recall@1"].tolist() == [1.0, 0.5]
    assert out["mrr"].tolist() == [1.0, 1.0]


def test_gate_needs_both_tasks_and_speed():
    rng = np.random.default_rng(0)
    champ = {"recall": rng.random(500), "abs_error": rng.random(500) * 10}
    better = {"recall": champ["recall"] + 0.1, "abs_error": champ["abs_error"] - 1, "p99_ms": 5}
    assert ev.gate(better, champ)["promote"]
    assert not ev.gate({**better, "abs_error": champ["abs_error"] + 1}, champ)["promote"]
    assert not ev.gate({**better, "p99_ms": 80}, champ)["promote"]


@pytest.mark.parametrize("n", [1, 7])
def test_bootstrap_interval_contains_mean(n):
    a = np.arange(100, dtype=float) * n
    m, lo, hi = ev.paired_bootstrap(a, a * 0.5)
    assert lo <= m <= hi

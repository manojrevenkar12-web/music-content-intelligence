# Model card: `mci-20261021-1738`

## What it does

For a new release (title, artist, album, Spotify's audio descriptors), it predicts:

1. **Which playlists suit it.** The 471 playlists are ranked by how likely each is to place the track.
   This is the playlist-pitching question.
2. **How popular it will be.** Spotify popularity, 0–100.

It is meant for a release team planning where to pitch new tracks and how much to invest in
them. It is not meant for judging artists, or for any decision about a single track without a
person in the loop.

## Data

TidyTuesday's 2020-01-21 release of Spotify Web API data: 28,352 tracks and 471 playlists, with
32,833 placements (CC0, collected with spotifyr in January 2020). Every run verifies the file
against a pinned SHA-256.

| Split | Released | Tracks | Used for |
|---|---|---:|---|
| train | before 2019-07-01 | 22,991 | fitting |
| validation | 2019-07-01 – 2019-09-30 | 1,774 | early stopping and choosing between candidates |
| test | from 2019-10-01 | 3,587 | the report and the gate, scored once |

The final models are refitted on train + validation (all history before the test period), with the
epoch count found on validation.

**Point-in-time inputs.** An artist's history is computed from releases strictly before the
track's release date. "Hit" is defined from training-period popularity only (top 20%, popularity
≥ 61). Track age at collection is an input, because popularity is a January 2020 snapshot and
new releases are mostly placed in new-music playlists.

**Known leak, stated.** An earlier release's popularity is its January 2020 value, so it
includes plays collected after the later release came out. This is the best point-in-time proxy
this dataset allows. A production feature store would snapshot popularity daily.

## Models

**Playlist matching** uses a two-tower network trained in PyTorch and served in ONNX Runtime.
- The track tower reads title and album words, the artist name, hashed character trigrams, the
  12 audio descriptors, and artist history and age.
- The playlist tower reads the playlist's name and a learned identity. The identity is dropped
  30% of the time in training, so a new playlist can be scored from its name alone.
- Word vectors start from spaCy's `en_core_web_md` (300-d, MIT) through a learned projection,
  plus a per-word correction that starts at zero.
- Loss: softmax over all playlists, plus an auxiliary popularity loss.

**Popularity** uses LightGBM on audio descriptors and artist history and age, converted to ONNX.

"Multimodal" here means text plus audio *descriptors*. Spotify computes the descriptors from the
audio. The model never sees waveforms or spectrograms: this dataset has none.

## Results on the test releases

Playlist recall@10 is the share of a track's real playlists that appear in the model's top 10.

| Playlist matching | recall@10 | MRR |
|---|---:|---:|
| recently active playlists | 0.091 | 0.051 |
| audio neighbours, recency-weighted | 0.187 | 0.096 |
| artist history + audio neighbours (strongest heuristic) | 0.293 | 0.202 |
| two-tower: audio + context only | 0.317 | 0.157 |
| two-tower: + text, no pretrained vectors | 0.307 | 0.174 |
| two-tower: + text, pretrained vectors frozen | 0.347 | 0.188 |
| **two-tower: + text, pretrained vectors fine-tuned (served)** | **0.340** | 0.184 |

- **Pretrained vectors are what make text useful.** Text without them does not beat audio alone
  on the test releases (0.307 vs 0.317).
- **Fine-tuning versus freezing is a tie.** On validation, fine-tuned won narrowly (0.361 vs
  0.360). On test, frozen did (0.347 vs 0.340). The served variant was chosen on validation, as
  it should be, and the difference is within noise.
- **The strongest heuristic still has the best MRR** (0.202 vs 0.184). When an artist has a
  history, "where they went before" is often exactly right at rank 1. The model wins on how many
  real playlists reach the top 10.

| Cold start (recall@10) | tracks | model | heuristic | gain, 95% CI |
|---|---:|---:|---:|---|
| debut artists | 1,800 | **0.238** | 0.177 | +0.061 [0.038, 0.084] |
| artists with history | 1,787 | 0.442 | 0.409 | +0.033 [0.010, 0.058] |

Half the test releases are by artists with no earlier track in the catalogue. That is where the
model adds most, because it reads the words and the sound, not the artist's past.

| Popularity | MAE | Spearman | hit AUC |
|---|---:|---:|---:|
| training mean | 15.74 | — | 0.500 |
| artist history (shrunk) | 13.46 | 0.483 | 0.765 |
| neural network, selected for popularity | 13.09 | 0.469 | 0.782 |
| mean of LightGBM and network | 11.02 | 0.585 | 0.838 |
| **LightGBM on audio + context (served, chosen on validation)** | **10.46** | **0.633** | **0.861** |

An earlier version trained one network for both tasks. It lost to LightGBM on popularity in that
run, and the release gate refused it. That is why each task now has its own champion.

## Release gate (`reports/training.json`)

The paired bootstrap runs over the 3,587 test releases, against the strongest heuristic on each task:

| Check | Result |
|---|---|
| playlist recall@10 better | +0.047, 95% CI [0.030, 0.062] ✓ |
| popularity error lower | 3.00 points better, 95% CI [2.71, 3.28] ✓ |
| single-request latency, int8 | p99 1.64 ms (limit 50 ms) ✓ |

The model is registered in MLflow as `mci-track-model` v1 with the alias `champion`.

## int8

Dynamic quantisation of the track model shrinks it from 13.3 MB to 3.3 MB. Recall@10 is 0.3399
for int8 and 0.3396 for fp32. The two precisions share 96% of their top-10 playlists.
Popularity is unchanged, because it comes from the LightGBM model.

## Limits

- **One snapshot, 2019 music.** Tastes, playlists and Spotify's popularity formula have moved
  on. Use the scores to rank options, and retrain on the label's own catalogue before relying on
  them.
- **Popularity is Spotify's index, not streams or revenue.** It rewards recent plays.
- **Playlists in this dataset were collected per genre**, so playlist recall rewards
  genre-consistent suggestions. Editorial playlists in the wild are more varied.
- **The English word vectors miss most Spanish and Portuguese words.** Character trigrams
  partly cover that, but a multilingual encoder is the obvious next step.
- **The online-learning results are semi-synthetic**: real tracks and real placements, with
  simulated listeners (`reports/online_learning.json`).

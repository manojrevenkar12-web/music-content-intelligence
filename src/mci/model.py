"""The multimodal track model (PyTorch, training only; serving runs the ONNX export).

Two towers and two heads:

    title + album words ─┐                                  ┌─► popularity (0-100)
    artist name words ───┤                                  │
    character trigrams ──┼─► track tower ─► track vector ───┤
    audio descriptors ───┤                    (64-d)        └─► · playlist vector ─► playlist scores
    artist history, age ─┘                                          ▲
                              playlist name words + id ─► playlist tower

Word vectors start from spaCy's pretrained English vectors (300-d) through a learned projection,
plus a per-word correction that starts at zero: the pretrained knowledge is kept and fine-tuned.
Hashed character trigrams carry the words those vectors do not cover.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mci.features import N_AUDIO, N_CONTEXT, TRI_BUCKETS

DIM = 64


def masked_mean(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    m = (ids > 0).unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class TextEncoder(nn.Module):
    def __init__(self, pretrained: np.ndarray, use_pretrained: bool = True, finetune: bool = True):
        super().__init__()
        v, d = pretrained.shape
        self.register_buffer("pretrained", torch.from_numpy(pretrained.astype(np.float32)), persistent=False)
        self.proj = nn.Linear(d, DIM, bias=False)
        self.delta = nn.Embedding(v, DIM, padding_idx=0)
        nn.init.zeros_(self.delta.weight)
        self.tris = nn.Embedding(TRI_BUCKETS, DIM, padding_idx=0)
        nn.init.normal_(self.tris.weight, std=0.05)
        self.use_pretrained = use_pretrained
        if not finetune:  # ablation: frozen pretrained vectors, no per-word correction
            self.delta.weight.requires_grad_(False)

    def word_table(self) -> torch.Tensor:
        """Every word's final vector: what the ONNX export stores as one table."""
        base = self.proj(self.pretrained) if self.use_pretrained else torch.zeros(self.delta.weight.shape, device=self.delta.weight.device)
        return base + self.delta.weight

    def forward(self, words: torch.Tensor, tris: torch.Tensor, table: torch.Tensor | None = None) -> torch.Tensor:
        table = self.word_table() if table is None else table
        return masked_mean(table[words], words) + masked_mean(self.tris(tris), tris)


class TrackModel(nn.Module):
    def __init__(self, pretrained: np.ndarray, n_playlists: int, use_text: bool = True, use_pretrained: bool = True, finetune: bool = True):
        super().__init__()
        self.use_text = use_text
        self.text = TextEncoder(pretrained, use_pretrained, finetune)
        self.audio = nn.Sequential(nn.Linear(N_AUDIO, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.track = nn.Sequential(nn.Linear(3 * DIM + DIM + N_CONTEXT, 2 * DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * DIM, 2 * DIM), nn.GELU())
        self.track_out = nn.Linear(2 * DIM, DIM)
        self.pop = nn.Sequential(nn.Linear(2 * DIM + N_CONTEXT, DIM), nn.GELU(), nn.Linear(DIM, 1))
        self.playlist_id = nn.Embedding(n_playlists, DIM)
        nn.init.normal_(self.playlist_id.weight, std=0.05)
        self.playlist = nn.Sequential(nn.Linear(2 * DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    # ------------------------------------------------------------------ towers
    def track_hidden(self, b: dict[str, torch.Tensor], table: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_text:
            t = self.text(b["title_words"], b["title_tris"], table)
            a = self.text(b["artist_words"], b["artist_tris"], table)
        else:
            t = a = torch.zeros(b["audio"].shape[0], DIM, device=b["audio"].device)
        au = self.audio(b["audio"])
        return self.track(torch.cat([t, a, t * au, au, b["context"]], dim=1))

    def track_vector(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.track_out(h), dim=1)

    def popularity(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return 100.0 * torch.sigmoid(self.pop(torch.cat([h, context], dim=1)).squeeze(1))

    def playlist_vectors(self, words: torch.Tensor, tris: torch.Tensor, drop_ids: float = 0.0) -> torch.Tensor:
        name = self.text(words, tris) if self.use_text else torch.zeros(words.shape[0], DIM, device=words.device)
        ids = self.playlist_id.weight
        if drop_ids > 0 and self.training:  # learn to score playlists from their names alone (new playlists)
            keep = (torch.rand(ids.shape[0], 1, device=ids.device) > drop_ids).to(ids.dtype)
            ids = ids * keep
        return nn.functional.normalize(self.playlist(torch.cat([name, ids], dim=1)), dim=1)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)


class TrackExport(nn.Module):
    """The serving graph: track inputs → (64-d track vector, popularity). The word table is folded
    into one constant, so the graph needs neither the 300-d vectors nor spaCy."""

    def __init__(self, m: TrackModel):
        super().__init__()
        self.m = m
        with torch.no_grad():
            self.register_buffer("table", m.text.word_table().detach().clone())

    def forward(self, audio, context, title_words, title_tris, artist_words, artist_tris):
        b = {"audio": audio, "context": context, "title_words": title_words, "title_tris": title_tris, "artist_words": artist_words, "artist_tris": artist_tris}
        h = self.m.track_hidden(b, self.table)
        return self.m.track_vector(h), self.m.popularity(h, context)

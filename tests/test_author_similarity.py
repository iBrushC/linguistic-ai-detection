"""Tests for src.author_similarity: aggregation, same-doc exclusion, normalization."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import author_similarity


def _perfect_cluster_dists(authors: list[str], docs: list[str], seed: int = 0) -> np.ndarray:
    """Same-author chunk pairs get distance 0.1; cross-author get 0.9."""
    n = len(authors)
    D = np.full((n, n), 0.9, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if authors[i] == authors[j] and docs[i] != docs[j]:
                D[i, j] = D[j, i] = 0.1
    np.fill_diagonal(D, 0.0)
    return D


def test_matrix_shape_and_authors_sorted():
    authors = ["B", "A", "B", "A"]
    docs = ["d1", "d2", "d3", "d4"]
    D = _perfect_cluster_dists(authors, docs)
    M, order = author_similarity.author_similarity_matrix(D, authors, docs)
    assert order == ["A", "B"]
    assert M.shape == (2, 2)
    assert M.min() >= 0.0 and M.max() <= 1.0


def test_diagonal_beats_offdiagonal_on_perfect_clusters():
    authors = ["A", "A", "A", "B", "B", "B"]
    docs = ["d1", "d2", "d3", "d4", "d5", "d6"]
    D = _perfect_cluster_dists(authors, docs)
    M, order = author_similarity.author_similarity_matrix(D, authors, docs)
    diag = np.diag(M)
    off = M[~np.eye(2, dtype=bool)]
    assert order == ["A", "B"]
    assert (diag > off).all(), f"diag={diag.tolist()} off={off.tolist()}"
    assert diag[0] == pytest.approx(1.0)
    assert diag[1] == pytest.approx(1.0)
    assert off[0] == pytest.approx(0.0)


def test_same_source_document_exclusion_drops_pairs():
    authors = ["A", "A", "B", "B"]
    docs = ["d1", "d1", "d2", "d3"]
    D = np.array([
        [0.0, 0.1, 0.5, 0.6],
        [0.1, 0.0, 0.5, 0.6],
        [0.5, 0.5, 0.0, 0.7],
        [0.6, 0.6, 0.7, 0.0],
    ])
    M_keep, _ = author_similarity.author_similarity_matrix(
        D, authors, docs, exclude_same_source_document=True,
    )
    M_all, _ = author_similarity.author_similarity_matrix(
        D, authors, docs, exclude_same_source_document=False,
    )
    # With exclusion the only A-A pair (d1, d1) is dropped, so the
    # (A, A) cell has zero pairs left and the cell is NaN.
    assert np.isnan(M_keep[0, 0])
    # Without exclusion the (A, A) cell distance is 0.1, which equals
    # the global off-diagonal minimum, so its normalised similarity is 1.0.
    assert M_all[0, 0] == pytest.approx(1.0)


def test_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    n = 6
    authors = ["A", "A", "B", "B", "C", "C"]
    docs = [f"d{i}" for i in range(n)]
    D = rng.uniform(0.0, 1.0, size=(n, n))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    M, _ = author_similarity.author_similarity_matrix(D, authors, docs)
    assert np.allclose(M, M.T)


def test_normalize_false_preserves_distance_sign():
    authors = ["A", "A", "B", "B"]
    docs = ["d1", "d2", "d3", "d4"]
    D = np.array([
        [0.0, 0.3, 0.9, 0.9],
        [0.3, 0.0, 0.9, 0.9],
        [0.9, 0.9, 0.0, 0.8],
        [0.9, 0.9, 0.8, 0.0],
    ])
    M_norm, _ = author_similarity.author_similarity_matrix(
        D, authors, docs, normalize=True,
    )
    M_raw, _ = author_similarity.author_similarity_matrix(
        D, authors, docs, normalize=False,
    )
    # Without normalization the (A,A) cell is 1 - 0.3 = 0.7.
    assert M_raw[0, 0] == pytest.approx(0.7)
    # With normalize=True the A-A self cell beats cross cells.
    assert M_norm[0, 0] > M_norm[0, 1]
    assert 0.0 <= M_norm.min() and M_norm.max() <= 1.0


def test_build_writes_outputs(tmp_path):
    import json
    from src import chunking

    rng = np.random.default_rng(0)
    vocab_a = "the cat sat on the mat".split()
    vocab_b = "she sells sea shells by the shore".split()
    essays = []
    for aid, (author, vocab) in enumerate([("A", vocab_a), ("B", vocab_b)]):
        for art_i in range(3):
            text = " ".join(rng.choice(vocab, size=2000)) + f". doc {author} {art_i}"
            essays.append({
                "author": author,
                "text": text,
                "config": "test",
                "split": "test",
                "article_id": f"test/{author}/{aid}/art{art_i}",
            })
    essays_path = str(tmp_path / "essays.json")
    with open(essays_path, "w", encoding="utf-8") as f:
        json.dump(essays, f)

    # Build a tiny distance matrix by hand (no pipeline).
    chunks = chunking.chunk_corpus(essays, chunk_size=200, min_fill_ratio=0.8)
    n = len(chunks)
    D = rng.uniform(0.0, 1.0, size=(n, n))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    distances_dir = tmp_path / "distances"
    distances_dir.mkdir()
    np.save(str(distances_dir / "cosine_delta.npy"), D)

    cfg = {
        "chunk_size_tokens": 200,
        "min_chunk_fill_ratio": 0.8,
        "min_chunks_per_author": 1,
        "essays_path": essays_path,
        "exclude_same_source_document": True,
    }

    out_dir = tmp_path / "out"
    summary = author_similarity.build(
        cfg,
        distances_dir=str(distances_dir),
        out_dir=str(out_dir),
        metrics=["cosine_delta"],
    )
    assert os.path.isfile(str(out_dir / "author_similarity_cosine_delta.png"))
    assert os.path.isfile(str(out_dir / "author_similarity_cosine_delta.json"))
    assert not os.path.isfile(str(out_dir / "author_similarity_combined.png"))
    matrix = summary["per_metric"]["cosine_delta"]["matrix"]
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
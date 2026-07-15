"""Distance metrics for Delta-style stylometric attribution.

All three metrics are effect-size-based: they measure *how far apart* two
text representations are, regardless of sample size. Two use a z-score
delta where the per-feature mean and standard deviation come from a
held-out reference corpus (frozen once per run, never re-fit on the test
chunks). The third uses tf-idf vectors directly.

* ``cosine_delta`` - cosine distance between z-scored vectors
* ``burrows_delta`` - Manhattan (L1) distance between z-scored vectors
* ``char_ngram_cosine`` - cosine distance between raw tf-idf vectors
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp


EPS = 1e-12


def zscore(X, mu, sigma, eps: float = EPS):
    """Z-score a dense or sparse matrix X against reference (mu, sigma).

    Columns whose ``sigma`` is zero or near-zero are left centered but
    left un-divided to avoid division-by-zero blow-ups.
    """
    if sp.issparse(X):
        X = X.astype(np.float64, copy=False)
        mu = np.asarray(mu, dtype=np.float64).ravel()
        sigma = np.asarray(sigma, dtype=np.float64).ravel()
        safe = np.where(sigma > eps, sigma, 1.0)
        centered = X - mu
        return centered.multiply(1.0 / safe)
    X = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    safe = np.where(sigma > eps, sigma, 1.0)
    return (X - mu) / safe


def _l2(X):
    if sp.issparse(X):
        return np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return np.linalg.norm(X, axis=1)


def cosine_distance(u, v) -> float:
    """Cosine distance = 1 - cos(u, v). Handles dense or sparse inputs."""
    if sp.issparse(u):
        u = u.toarray().ravel()
    if sp.issparse(v):
        v = v.toarray().ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return 0.0
    sim = float(np.dot(u, v) / (nu * nv))
    return float(1.0 - sim)


def cosine_distance_matrix(Xz: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance over a (n, d) dense matrix.

    Returned matrix is symmetric with zeros on the diagonal; values in
    [0, 2] but typically [0, 1] in practice.
    """
    Xz = np.asarray(Xz, dtype=np.float64)
    norms = _l2(Xz)
    safe = np.where(norms > EPS, norms, 1.0)
    Xn = Xz / safe[:, None]
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def manhattan_distance_matrix(Xz: np.ndarray) -> np.ndarray:
    """Pairwise L1 distance over a (n, d) dense matrix.

    Diagonal is zero; otherwise symmetric.
    """
    Xz = np.asarray(Xz, dtype=np.float64)
    diffs = np.abs(Xz[:, None, :] - Xz[None, :, :]).sum(axis=2)
    np.fill_diagonal(diffs, 0.0)
    return diffs


def sparse_cosine_distance_matrix(X) -> np.ndarray:
    """Pairwise cosine distance between sparse tf-idf rows.

    Computes (X * X.T) row-wise, normalises by row norms.
    """
    if not sp.issparse(X):
        return cosine_distance_matrix(np.asarray(X))
    X = X.tocsr().astype(np.float64)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    safe = np.where(norms > EPS, norms, 1.0)
    inv = sp.diags(1.0 / safe)
    Xn = inv @ X
    sim = (Xn @ Xn.T).toarray()
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def cosine_delta_matrix(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Cosine distance between z-scored vectors."""
    Xz = zscore(X, mu, sigma)
    return cosine_distance_matrix(Xz)


def burrows_delta_matrix(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Manhattan (L1) distance between z-scored vectors.

    This is the original Burrows (2002) Delta on z-scored word-frequency
    vectors.
    """
    Xz = zscore(X, mu, sigma)
    return manhattan_distance_matrix(Xz)


def char_ngram_distance_matrix(X) -> np.ndarray:
    """Cosine distance on tf-idf vectors without any additional z-scoring.

    tf-idf already normalises within a chunk and reweights by document
    rarity across the union corpus.
    """
    return sparse_cosine_distance_matrix(X)


__all__ = [
    "EPS",
    "zscore",
    "cosine_distance",
    "cosine_distance_matrix",
    "manhattan_distance_matrix",
    "sparse_cosine_distance_matrix",
    "cosine_delta_matrix",
    "burrows_delta_matrix",
    "char_ngram_distance_matrix",
]

"""Tests for src.evaluation: AUC, permutation p-value, same-source-doc exclusion."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import evaluation


def _toy_distance(n: int = 8, seed: int = 0) -> np.ndarray:
    """Build a perfect-clustering distance matrix: self blocks near zero."""
    rng = np.random.default_rng(seed)
    D = rng.uniform(0.5, 1.0, size=(n, n))
    # Two clusters: 0..3 author A, 4..7 author B
    D[:4, :4] = rng.uniform(0.0, 0.05, size=(4, 4))
    D[4:, 4:] = rng.uniform(0.0, 0.05, size=(4, 4))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def test_pair_labels_excludes_same_document():
    authors = ["A", "A", "B", "B"]
    docs = ["d1", "d1", "d2", "d3"]
    D = np.array([
        [0.0, 0.1, 0.7, 0.8],
        [0.1, 0.0, 0.6, 0.5],
        [0.7, 0.6, 0.0, 0.9],
        [0.8, 0.5, 0.9, 0.0],
    ])
    pair_d, is_self = evaluation.pair_labels(D, authors, docs)
    # Three authors-A (i=0,j=1) keep if same doc — wait: doc1==doc1 so dropped
    # So kept pairs: (0,2) cross, (0,3) cross, (1,2) cross, (1,3) cross, (2,3) self
    assert len(pair_d) == 5
    assert int(is_self.sum()) == 1
    assert int((1 - is_self).sum()) == 4


def test_pair_labels_keeps_different_doc_self():
    authors = ["A", "A", "B", "B"]
    docs = ["d1", "d2", "d3", "d4"]
    D = np.array([
        [0.0, 0.1, 0.7, 0.8],
        [0.1, 0.0, 0.6, 0.5],
        [0.7, 0.6, 0.0, 0.9],
        [0.8, 0.5, 0.9, 0.0],
    ])
    pair_d, is_self = evaluation.pair_labels(D, authors, docs)
    assert int(is_self.sum()) == 2  # (0,1) author A + (2,3) author B
    assert int((1 - is_self).sum()) == 4


def test_auc_perfect_separation_is_one():
    pair_d = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    is_self = np.array([1, 1, 0, 0, 0])
    auc = evaluation.auc_self_vs_cross(pair_d, is_self)
    assert auc == pytest.approx(1.0)


def test_auc_random_is_near_half():
    rng = np.random.default_rng(7)
    n = 200
    pair_d = rng.normal(size=n)
    is_self = rng.integers(0, 2, size=n)
    auc = evaluation.auc_self_vs_cross(pair_d, is_self)
    assert 0.3 < auc < 0.7


def test_auc_singleton_class_is_nan():
    auc = evaluation.auc_self_vs_cross(np.array([0.1, 0.2]), np.array([1, 1]))
    assert np.isnan(auc)


def test_permutation_pvalue_strong_signal():
    D = _toy_distance()
    authors = ["A"] * 4 + ["B"] * 4
    docs = [f"d{i}" for i in range(8)]
    iu, ju, _ = evaluation._build_pair_labels(authors, docs, exclude_same_source_document=True)
    pair_d, is_self = evaluation.pair_labels(D, authors, docs)
    out = evaluation.permutation_auc(
        authors, distances=pair_d, is_self=is_self,
        iu=iu, ju=ju, n_permutations=200, rng_seed=1729,
    )
    assert out["observed_auc"] == pytest.approx(1.0)
    assert out["p_value"] < 0.05


def test_permutation_pvalue_no_signal():
    D = np.full((8, 8), 0.5)
    np.fill_diagonal(D, 0.0)
    authors = ["A"] * 4 + ["B"] * 4
    docs = [f"d{i}" for i in range(8)]
    iu, ju, _ = evaluation._build_pair_labels(authors, docs, exclude_same_source_document=True)
    pair_d, is_self = evaluation.pair_labels(D, authors, docs)
    out = evaluation.permutation_auc(
        authors, distances=pair_d, is_self=is_self,
        iu=iu, ju=ju, n_permutations=100, rng_seed=1729,
    )
    assert 0.3 < out["observed_auc"] < 0.7
    assert out["p_value"] > 0.05

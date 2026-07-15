"""Tests for src.distance z-scoring and metric matrices."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import distance


def test_zscore_basic():
    X = np.array([[10.0, 20.0], [12.0, 18.0], [8.0, 22.0]])
    mu = np.array([10.0, 20.0])
    sigma = np.array([2.0, 2.0])
    Z = distance.zscore(X, mu, sigma)
    expected = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(Z, expected)


def test_zscore_handles_zero_sigma():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.array([2.0, 3.0])
    sigma = np.array([0.0, 1.0])
    Z = distance.zscore(X, mu, sigma)
    assert not np.isnan(Z).any()
    # First column is constant-centered (no division-by-zero)
    assert Z[0, 0] == -1.0
    assert Z[1, 0] == 1.0
    # Second column is normally divided
    assert np.allclose(Z[:, 1], [-1.0, 1.0])


def test_cosine_distance_self_is_zero():
    v = np.array([1.0, 2.0, 3.0, 0.0])
    assert distance.cosine_distance(v, v) == pytest.approx(0.0)


def test_cosine_distance_orthogonal_is_one():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert distance.cosine_distance(a, b) == pytest.approx(1.0)


def test_cosine_distance_matrix_diagonal_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    D = distance.cosine_distance_matrix(X)
    assert D.shape == (10, 10)
    assert np.allclose(D, D.T)
    np.testing.assert_allclose(np.diag(D), np.zeros(10))


def test_cosine_distance_matrix_symmetric_non_negative():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    D = distance.cosine_distance_matrix(X)
    assert (D >= -1e-10).all()
    assert np.allclose(D, D.T)


def test_manhattan_distance_matrix_diagonal_zero():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 3))
    D = distance.manhattan_distance_matrix(X)
    assert D.shape == (8, 8)
    assert np.allclose(np.diag(D), np.zeros(8))
    assert np.allclose(D, D.T)


def test_cosine_delta_uses_reference_not_self():
    """Cosine Delta must use the held-out reference mu/sigma, not the test stats."""
    rng = np.random.default_rng(3)
    X = rng.normal(loc=5.0, scale=1.0, size=(4, 6))
    ref_mu = np.zeros(6)
    ref_sigma = np.ones(6)
    D_ref = distance.cosine_delta_matrix(X, ref_mu, ref_sigma)
    test_mu = X.mean(axis=0)
    test_sigma = X.std(axis=0, ddof=0)
    D_self = distance.cosine_delta_matrix(X, test_mu, test_sigma)
    assert not np.allclose(D_ref, D_self)
    assert D_ref[0, 1] != pytest.approx(D_self[0, 1])


def test_burrows_delta_symmetric():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 4))
    mu = np.zeros(4)
    sigma = np.ones(4)
    D = distance.burrows_delta_matrix(X, mu, sigma)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), np.zeros(5))

"""Tests for src.features: MFW extraction, char n-grams, reference stats."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import MFWExtractor, CharNGramExtractor


SAMPLE_TEXTS = [
    "the the the cat cat sat on the mat by the door to the house",
    "she she she sells sea shells by the sea shore every morning",
    "the cat sat on the mat by the door and looked at the sea",
    "sells sells sells sea shells by the shore from the morning boat",
    "the dog the dog the dog ran to the door of the house",
]


def test_mfw_vocab_size_capped_at_n():
    with pytest.raises(ValueError):
        MFWExtractor(n=20)  # outside allowed range
    ex = MFWExtractor(n=100)
    ex.fit(SAMPLE_TEXTS)
    assert len(ex.vocab_) <= 100
    assert len(ex.vocab_) > 0
    assert "the" in ex.vocab_


def test_mfw_vector_rows_sum_to_one():
    ex = MFWExtractor(n=100)
    ex.fit(SAMPLE_TEXTS)
    X = ex.transform(SAMPLE_TEXTS)
    sums = X.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(X.shape[0]), atol=1e-9)


def test_mfw_reference_stats_match_input():
    ex = MFWExtractor(n=100)
    ex.fit(SAMPLE_TEXTS)
    stats = ex.reference_stats
    assert stats is not None
    mu = np.asarray(stats["mu"])
    sigma = np.asarray(stats["sigma"])
    assert mu.shape == (len(ex.vocab_),)
    assert sigma.shape == (len(ex.vocab_),)
    assert (sigma >= 0).all()
    j_the = ex.vocab_.index("the")
    assert mu[j_the] == mu.max()


def test_mfw_n_out_of_range_rejected():
    with pytest.raises(ValueError):
        MFWExtractor(n=50)
    with pytest.raises(ValueError):
        MFWExtractor(n=2000)


def test_char_ngram_shape():
    ex = CharNGramExtractor(ngram_range=(3, 4), min_df=2)
    X = ex.fit_transform(SAMPLE_TEXTS)
    assert X.shape[0] == len(SAMPLE_TEXTS)
    assert X.shape[1] > 0
    # tf-idf should not be all zeros for short inputs
    assert X.nnz > 0


def test_char_ngram_no_zscore_required():
    ex = CharNGramExtractor()
    assert ex.needs_zscoring is False


def test_char_ngram_keeps_whitespace_and_punctuation():
    ex = CharNGramExtractor(ngram_range=(3, 3), min_df=1)
    ex.fit(["hello world!"])
    names = ex.get_feature_names()
    # char_wb emits boundary ' ' + word chars, so punctuation should show up at boundaries
    assert any("!" in n for n in names)


def test_mfw_needs_zscore_true():
    assert MFWExtractor().needs_zscoring is True

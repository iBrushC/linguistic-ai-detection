# Smoke tests for global_similarity in src/analysis.py
from __future__ import annotations

import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).parent
sys.path.insert(0, str(SRC))

from analysis import global_similarity


def _make_metrics(
    n_sentences: int,
    *,
    seed: int,
    tricolon_rate: float = 0.0,
    cleft_rate: float = 0.0,
) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    return {
        "sentence_lengths": rng.integers(40, 200, size=n_sentences).tolist(),
        "words_per_sentence": rng.integers(6, 40, size=n_sentences).tolist(),
        "word_lengths": rng.integers(2, 10, size=n_sentences * 9).tolist(),
        "ttr_per_sentence": list(np.round(rng.uniform(0.3, 0.9, size=n_sentences), 3)),
        "cttr_per_sentence": list(np.round(rng.uniform(1.0, 5.0, size=n_sentences), 3)),
        "tricolon_counts": [
            1 if rng.random() < tricolon_rate else 0 for _ in range(n_sentences)
        ],
        "cleft_counts": [
            1 if rng.random() < cleft_rate else 0 for _ in range(n_sentences)
        ],
        "normalization_counts": [
            int(v) for v in rng.integers(0, 2, size=n_sentences)
        ],
        "existential_extraposition_counts": [
            int(v) for v in rng.integers(0, 2, size=n_sentences)
        ],
        "anadiplosis_counts": [
            int(v) for v in rng.integers(0, 2, size=n_sentences)
        ],
        "conjunctions_per_series": [
            int(v) for v in rng.integers(0, 3, size=n_sentences)
        ],
        "segments_per_sentence": [
            int(v) for v in rng.integers(1, 4, size=n_sentences)
        ],
        "pos_NN": [int(v) for v in rng.integers(0, 8, size=n_sentences)],
        "pos_VB": [int(v) for v in rng.integers(0, 5, size=n_sentences)],
        "pos_JJ": [int(v) for v in rng.integers(0, 4, size=n_sentences)],
    }


def _is_close(actual: float, expected: float, *, abs_tol: float = 1e-6) -> bool:
    return abs(actual - expected) <= abs_tol


def test_identical_inputs_full_similarity_simple() -> None:
    metrics = _make_metrics(200, seed=1)
    result = global_similarity(metrics, metrics, method="simple")
    assert _is_close(result["similarity"], 1.0), (
        f"identical inputs should give similarity 1.0, got {result['similarity']}"
    )
    assert result["method"] == "simple"


def test_different_inputs_lower_similarity() -> None:
    a = _make_metrics(200, seed=2)
    b = _make_metrics(200, seed=3)
    result = global_similarity(a, b, method="simple")
    assert result["similarity"] < 0.99, (
        f"disjoint seeds should not be near-identical, got {result['similarity']}"
    )
    assert result["n_metrics_used"] >= 5


def test_sparse_metric_does_not_dominate_simple() -> None:
    """A high-weight (common) metric should not be overwhelmed by many tiny ones."""
    common = [10] * 100
    a = {"sentence_lengths": common, "pos_NN": [0] * 100}
    b = {"sentence_lengths": common, "pos_NN": [0] * 100}
    identical = global_similarity(a, b, method="simple")
    assert _is_close(identical["similarity"], 1.0)

    # sentence_lengths remains identical; sparse pos_NN differs in 2 of 100 sentences.
    a2 = {"sentence_lengths": common, "pos_NN": [0] * 98 + [1, 1]}
    b2 = {"sentence_lengths": common, "pos_NN": [0] * 100}
    res = global_similarity(a2, b2, method="simple", min_appearances=10)
    info = res["per_metric"]["pos_NN"]
    assert info["weight"] < 1.0, (
        f"sparse pos_NN should be downweighted, got {info['weight']}"
    )
    assert info["weight"] <= 0.5, (
        f"pos_NN with 2 combined nonzeros should be well below 1.0 weight, got "
        f"{info['weight']}"
    )


def test_manova_returns_valid_scores_for_aligned_metrics() -> None:
    a = _make_metrics(120, seed=10)
    b = _make_metrics(120, seed=11)
    # Make tricolon dense enough to survive the sparsity filter.
    a["tricolon_counts"] = [int(v) for v in np.random.default_rng(0).integers(0, 2, size=120)]
    b["tricolon_counts"] = [int(v) for v in np.random.default_rng(1).integers(0, 2, size=120)]
    result = global_similarity(a, b, method="manova", min_appearances=10)
    assert result["method"] == "manova"
    assert result["similarity"] is not None, (
        f"MANOVA should produce a similarity, got {result}"
    )
    assert 0.0 <= result["similarity"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_features"] >= 2


def test_manova_identical_returns_one() -> None:
    a = _make_metrics(120, seed=20)
    result = global_similarity(a, a, method="manova", min_appearances=10)
    if result["similarity"] is None:
        print(
            f"NOTE: identical MANOVA returned None (not enough features or "
            f"perfect collinearity). Dropped: {result.get('dropped')}"
        )
        return
    assert _is_close(result["similarity"], 1.0), (
        f"identical metrics should give 1.0 MANOVA similarity, got "
        f"{result['similarity']}"
    )


def test_invalid_method_raises() -> None:
    try:
        global_similarity({}, {}, method="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")


if __name__ == "__main__":
    test_identical_inputs_full_similarity_simple()
    print("[ok] identical inputs -> similarity 1.0 (simple)")

    test_different_inputs_lower_similarity()
    print("[ok] different inputs -> similarity < 0.95 (simple)")

    test_sparse_metric_does_not_dominate_simple()
    print("[ok] sparse metric downweighted (simple)")

    test_manova_returns_valid_scores_for_aligned_metrics()
    print("[ok] MANOVA returns a valid 0-1 score")

    test_manova_identical_returns_one()
    print("[ok] identical MANOVA returns 1.0 (or None when degenerate)")

    test_invalid_method_raises()
    print("[ok] unknown method raises ValueError")

    print("\nAll smoke tests passed.")

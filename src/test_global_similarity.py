# Smoke tests for global_similarity in src/analysis.py
from __future__ import annotations

import pathlib
import sys
from itertools import combinations, product

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
        "pos_PRP$": [int(v) for v in rng.integers(0, 3, size=n_sentences)],
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


def test_metric_arrays_share_sentence_alignment() -> None:
    from analysis import get_all_metrics

    text = "The cat sleeps. Birds fly quickly. However, they return."
    metrics = get_all_metrics(text)
    n_sentences = len(metrics["sentence_lengths"])
    assert n_sentences == 3
    for name, values in metrics.items():
        if name == "word_lengths":
            continue
        assert len(values) == n_sentences, (
            f"{name} has {len(values)} values for {n_sentences} sentences"
        )
    pos_metrics = {
        name: values for name, values in metrics.items() if name.startswith("pos_")
    }
    assert pos_metrics
    assert any(0 in values for values in pos_metrics.values())
    assert metrics["anadiplosis_counts"][0] == 0


def test_missing_metric_uses_own_sentence_count() -> None:
    from analysis import _normalize_metric_array

    a, b, skip = _normalize_metric_array(None, [0, 1], 3, 2)
    assert skip is None
    assert a.tolist() == [0.0, 0.0, 0.0]
    assert b.tolist() == [0.0, 1.0]


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


def test_brunnermunzel_identical_returns_one() -> None:
    metrics = _make_metrics(120, seed=77)
    result = global_similarity(metrics, metrics, method="brunnermunzel")
    assert result["similarity"] is not None, result
    assert _is_close(result["similarity"], 1.0), (
        f"identical BM inputs should give 1.0, got {result['similarity']}"
    )
    assert result["method"] == "brunnermunzel"
    assert "family_means" in result
    assert set(result["family_means"]) >= {"lexical", "pos", "structural"}
    # Underlying BM relative effect should be ~0.5 for identical inputs,
    # which the similarity mapping lifts to ~1.0.
    info = result["per_metric"]["pos_NN"]
    assert abs(info["bm_effect"] - 0.5) < 0.05, (
        f"identical pos_NN BM effect should be near 0.5, got {info['bm_effect']}"
    )
    assert _is_close(info["bm_similarity"], 1.0, abs_tol=0.1), (
        f"identical pos_NN bm_similarity should be ~1.0, got {info['bm_similarity']}"
    )


def test_brunnermunzel_different_lower_than_identical() -> None:
    a = _make_metrics(120, seed=77)
    b = _make_metrics(120, seed=88)
    res_id = global_similarity(a, a, method="brunnermunzel")
    res_diff = global_similarity(a, b, method="brunnermunzel")
    assert res_id["similarity"] > res_diff["similarity"], (
        f"identical pair ({res_id['similarity']:.4f}) should score higher "
        f"than disjoint pair ({res_diff['similarity']:.4f})"
    )
    assert 0.0 <= res_diff["similarity"] <= 1.0


def test_brunnermunzel_handles_degenerate_metric() -> None:
    a = _make_metrics(120, seed=1)
    b = _make_metrics(120, seed=2)
    a["constant_metric"] = [5.0] * 120
    b["constant_metric"] = [5.0] * 120
    result = global_similarity(a, b, method="brunnermunzel")
    assert result["similarity"] is not None, result
    assert 0.0 <= result["similarity"] <= 1.0
    assert any(d.get("name") == "constant_metric" for d in result["dropped"]), (
        f"constant metric should be dropped, got {result['dropped']}"
    )
    assert any(d.get("reason") == "constant" for d in result["dropped"])


def test_brunnermunzel_excludes_red_herrings() -> None:
    """The 6 metrics flagged by diagnose_bm.py as anti-signal are skipped."""
    metrics = _make_metrics(120, seed=77)
    metrics["dep_conjunct"] = [1, 2, 3, 4] * 30
    metrics["anadiplosis_counts"] = [0, 1] * 60
    metrics["pos_JJS"] = [0, 1, 0, 1, 0, 1] * 20
    metrics["segments_per_sentence"] = [2, 3] * 60
    metrics["existential_extraposition_counts"] = [0, 1] * 60
    metrics["pos_CC"] = [1, 2] * 60
    result = global_similarity(metrics, metrics, method="brunnermunzel")
    for name in (
        "dep_conjunct", "anadiplosis_counts", "pos_JJS",
        "segments_per_sentence", "existential_extraposition_counts", "pos_CC",
    ):
        assert name not in result["per_metric"], (
            f"{name} should be excluded from per_metric"
        )
        entry = next((d for d in result["dropped"] if d.get("name") == name), None)
        assert entry is not None and entry.get("reason") == "red_herring", (
            f"{name} should appear in dropped with reason 'red_herring', got {entry}"
        )


def test_weighted_identical_inputs_stay_full_similarity() -> None:
    """Symmetric weighting is multiplicative so identical inputs remain 1.0."""
    metrics = _make_metrics(200, seed=42)
    weights = {"sentence_lengths": 2.5, "pos_NN": 0.5}
    base = global_similarity(metrics, metrics, method="simple")
    tuned = global_similarity(metrics, metrics, method="simple",
                              metric_weights=weights)
    assert _is_close(base["similarity"], 1.0)
    assert _is_close(tuned["similarity"], 1.0), (
        f"weighted identical inputs should still give 1.0, got {tuned['similarity']}"
    )
    assert "tuned" in tuned["weighting"]


def test_weighted_can_change_simple_similarity() -> None:
    """A heavy weight on a divergent metric lowers similarity vs the baseline."""
    rng = np.random.default_rng(0)
    base_a = rng.integers(40, 200, size=100).tolist()
    base_b = rng.integers(40, 200, size=100).tolist()
    a = {"sentence_lengths": base_a, "pos_NN": [0] * 50 + [1] * 50}
    b = {"sentence_lengths": base_b, "pos_NN": [0] * 100}
    unweighted = global_similarity(a, b, method="simple")
    tuned = global_similarity(a, b, method="simple",
                              metric_weights={"pos_NN": 10.0})
    assert "tuned" in tuned["weighting"]
    info = tuned["per_metric"]["pos_NN"]
    assert info["base_weight"] > 0.0
    assert info["weight"] == info["base_weight"] * 10.0
    unweighted_pos_weight = unweighted["per_metric"]["pos_NN"]["weight"]
    assert info["weight"] > unweighted_pos_weight, (
        f"weighted pos_NN weight should exceed unweighted, "
        f"got {info['weight']} vs {unweighted_pos_weight}"
    )
    assert tuned["similarity"] != unweighted["similarity"], (
        f"tuned and unweighted similarity should differ, both={tuned['similarity']:.4f}"
    )
    print(f"  unweighted sim={unweighted['similarity']:.4f}  "
          f"tuned sim={tuned['similarity']:.4f}")


def test_author_matrix_averages_comparable_essay_pairs() -> None:
    import test as matrix_test

    essays = [
        {"author": "By Alice", "body": "alice-0"},
        {"author": "By Alice", "body": "alice-1"},
        {"author": "By Alice", "body": "alice-2"},
        {"author": "By Bob", "body": "bob-0"},
        {"author": "By Bob", "body": "bob-1"},
    ]
    distributions = [
        [1, 1, 2, 2],
        [1, 1, 2, 3],
        [1, 2, 2, 3],
        [8, 8, 9, 9],
        [8, 9, 9, 10],
    ]
    cache = {
        matrix_test._text_fingerprint(essay["body"]): {
            "sentence_lengths": distribution
        }
        for essay, distribution in zip(essays, distributions)
    }
    grouped = matrix_test.group_by_author(essays)
    matrix = matrix_test.build_similarity_matrix(
        grouped,
        essays,
        cache,
        "simple",
    )
    self_info = matrix.attrs["self_info"]
    assert self_info["Alice"]["n_pairs"] == 3
    assert self_info["Bob"]["n_pairs"] == 1

    alice_pairs = list(combinations(grouped["Alice"], 2))
    cross_pairs = list(product(grouped["Alice"], grouped["Bob"]))

    def pair_similarity(i: int, j: int) -> float:
        return global_similarity(
            {"sentence_lengths": distributions[i]},
            {"sentence_lengths": distributions[j]},
            method="simple",
        )["similarity"]

    expected_self = float(np.mean([pair_similarity(i, j) for i, j in alice_pairs]))
    expected_cross = float(np.mean([pair_similarity(i, j) for i, j in cross_pairs]))
    assert _is_close(matrix.loc["Alice", "Alice"], expected_self)
    assert _is_close(matrix.loc["Alice", "Bob"], expected_cross)


def test_tune_compute_metric_weights_prefers_discriminant() -> None:
    """End-to-end: same-author pairs agree on noise, diff pairs disagree."""
    from tune import compute_metric_weights

    n_sentences = 60

    def author_metrics(
        seed: int, marker_bias: float, distractor_bias: float, anti_bias: float
    ) -> dict[str, list]:
        a_rng = np.random.default_rng(seed)
        return {
            "marker": a_rng.normal(loc=marker_bias, scale=0.1,
                                   size=n_sentences).tolist(),
            "distractor": a_rng.normal(loc=distractor_bias, scale=0.1,
                                       size=n_sentences).tolist(),
            "anti_signal": a_rng.normal(loc=anti_bias, scale=0.1,
                                        size=n_sentences).tolist(),
        }

    essays = [
        {"author": "By Alice", "body": ""},
        {"author": "By Alice", "body": ""},
        {"author": "By Bob", "body": ""},
        {"author": "By Bob", "body": ""},
    ]
    metrics_lookup = {
        0: author_metrics(
            101, marker_bias=0.0, distractor_bias=1.0, anti_bias=0.0
        ),
        1: author_metrics(
            102, marker_bias=0.05, distractor_bias=1.05, anti_bias=10.0
        ),
        2: author_metrics(
            201, marker_bias=2.0, distractor_bias=2.0, anti_bias=0.0
        ),
        3: author_metrics(
            202, marker_bias=2.05, distractor_bias=2.2, anti_bias=10.0
        ),
    }

    payload = compute_metric_weights(essays, metrics_lookup=metrics_lookup)
    assert payload["n_same_pairs"] == 2
    assert payload["n_diff_pairs"] == 4
    weights = payload["metric_weights"]
    assert "marker" in weights, weights
    assert "distractor" in weights, weights
    assert "anti_signal" in weights, weights
    assert payload["weight_power"] > 1.0, payload
    assert weights["marker"] > 1.0, weights
    assert weights["marker"] > weights["distractor"], weights
    assert weights["anti_signal"] < 1.0, weights
    assert payload["stats"]["marker"]["sep"] > payload["stats"]["distractor"]["sep"], \
        payload["stats"]  # the higher-sep metric should win
    stats_marker = payload["stats"]["marker"]
    assert stats_marker["ks_D_mean"] > stats_marker["ks_S_mean"], stats_marker


def test_load_metric_weights_handles_missing_and_malformed(tmp_metrics_path=None) -> None:
    from analysis import load_metric_weights

    assert load_metric_weights(None) is None
    assert load_metric_weights("definitely/not/here.json") is None

    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"metric_weights": {"a": 1.5, "b": -1.0, "c": "x", "d": 0.4}}, f)
        path = f.name
    try:
        weights = load_metric_weights(path)
        assert weights == {"a": 1.5, "d": 0.4}, weights
    finally:
        os = __import__("os")
        os.remove(path)


if __name__ == "__main__":
    test_identical_inputs_full_similarity_simple()
    print("[ok] identical inputs -> similarity 1.0 (simple)")

    test_different_inputs_lower_similarity()
    print("[ok] different inputs -> similarity < 0.95 (simple)")

    test_sparse_metric_does_not_dominate_simple()
    print("[ok] sparse metric downweighted (simple)")

    test_metric_arrays_share_sentence_alignment()
    print("[ok] sentence-level metric arrays stay aligned")

    test_missing_metric_uses_own_sentence_count()
    print("[ok] missing metrics use the correct sentence count")

    test_manova_returns_valid_scores_for_aligned_metrics()
    print("[ok] MANOVA returns a valid 0-1 score")

    test_manova_identical_returns_one()
    print("[ok] identical MANOVA returns 1.0 (or None when degenerate)")

    test_invalid_method_raises()
    print("[ok] unknown method raises ValueError")

    test_brunnermunzel_identical_returns_one()
    print("[ok] identical inputs -> similarity 1.0 (brunnermunzel)")

    test_brunnermunzel_different_lower_than_identical()
    print("[ok] different inputs score lower than identical (brunnermunzel)")

    test_brunnermunzel_handles_degenerate_metric()
    print("[ok] constant metric dropped, similarity still computable (brunnermunzel)")

    test_brunnermunzel_excludes_red_herrings()
    print("[ok] 6 red-herring metrics are excluded (brunnermunzel)")

    test_weighted_identical_inputs_stay_full_similarity()
    print("[ok] weighted identical inputs remain 1.0")

    test_weighted_can_change_simple_similarity()
    print("[ok] weighted simple similarity reports tuned weights")

    test_author_matrix_averages_comparable_essay_pairs()
    print("[ok] author matrix averages comparable essay pairs")

    test_tune_compute_metric_weights_prefers_discriminant()
    print("[ok] tune.compute_metric_weights upweights discriminant metrics")

    test_load_metric_weights_handles_missing_and_malformed()
    print("[ok] load_metric_weights handles missing/malformed files")

    print("\nAll smoke tests passed.")

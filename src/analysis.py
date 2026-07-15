# Analysis of stylometry on general works of text

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from statsmodels.multivariate.manova import MANOVA

from stylometry import (
    get_anadiplosis_counts,
    get_cleft_counts,
    get_conjunctions_per_series,
    get_connective_density,
    get_cttr_per_sentence,
    get_existential_extraposition_counts,
    get_normalization_counts,
    get_segments_per_sentence,
    get_sentence_lengths,
    get_syntactic_markers,
    get_tricolon_counts,
    get_ttr_per_sentence,
    get_word_lengths,
    get_word_types_per_sentence,
    get_words_per_sentence,
)


def _metric_summary(values) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std, int(arr.size)


def get_all_metrics(text: str) -> dict[str, list]:
    """Compute every stylometric marker in stylometry.py for the given text.

    Returns a flat dict mapping metric name to a list of per-sentence (or
    per-word) values. Compound metrics (POS tags, syntactic dependencies,
    connectives) are flattened with prefixed names so each entry is a single
    comparable distribution.
    """
    metrics: dict[str, list] = {
        "sentence_lengths": get_sentence_lengths(text),
        "words_per_sentence": get_words_per_sentence(text),
        "word_lengths": get_word_lengths(text),
        "ttr_per_sentence": get_ttr_per_sentence(text),
        "cttr_per_sentence": get_cttr_per_sentence(text),
        "tricolon_counts": get_tricolon_counts(text),
        "cleft_counts": get_cleft_counts(text),
        "normalization_counts": get_normalization_counts(text),
        "existential_extraposition_counts": get_existential_extraposition_counts(text),
        "anadiplosis_counts": get_anadiplosis_counts(text),
        "conjunctions_per_series": get_conjunctions_per_series(text),
        "segments_per_sentence": get_segments_per_sentence(text),
    }

    for tag, counts in get_word_types_per_sentence(text).items():
        metrics[f"pos_{tag}"] = counts

    for name, counts in get_syntactic_markers(text).items():
        metrics[f"dep_{name}"] = counts

    for name, counts in get_connective_density(text).items():
        metrics[f"connective_{name}"] = counts

    return metrics


def compare_metrics(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, dict]:
    """Run a two-tailed independent t-test on every metric present in both dicts.

    Each metric's per-sentence distribution is compared between the two
    corpora using scipy.stats.ttest_ind. Metrics that appear in only one
    corpus, have fewer than two samples in either group, or are constant in
    both groups are skipped.

    Returns a dict mapping metric name to:
        {
            "t_stat", "p_value",
            "mean_a", "std_a", "n_a",
            "mean_b", "std_b", "n_b",
        }
    """
    results: dict[str, dict] = {}
    for name in metrics_a:
        if name not in metrics_b:
            continue
        a = np.asarray(metrics_a[name], dtype=float)
        b = np.asarray(metrics_b[name], dtype=float)
        if a.size < 2 or b.size < 2:
            continue
        if np.std(a) == 0 and np.std(b) == 0:
            continue
        t_stat, p_val = ttest_ind(a, b, equal_var=False)
        mean_a, std_a, n_a = _metric_summary(a)
        mean_b, std_b, n_b = _metric_summary(b)
        results[name] = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "mean_a": mean_a,
            "std_a": std_a,
            "n_a": n_a,
            "mean_b": mean_b,
            "std_b": std_b,
            "n_b": n_b,
        }
    return results


def print_comparison(
    results: dict[str, dict],
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Pretty-print the comparison table: per-metric mean±std for each group
    plus t-statistic and p-value. Asterisk marks p < 0.05.
    """
    a_hdr = f"{label_a} mean+/-std"
    b_hdr = f"{label_b} mean+/-std"
    header = (
        f"{'Metric':<38} "
        f"{a_hdr:>22} "
        f"{b_hdr:>22} "
        f"{'t':>9} "
        f"{'p':>10} "
        f"{'sig':>4}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(results):
        r = results[name]
        sig = "*" if r["p_value"] < 0.05 else ""
        a_str = f"{r['mean_a']:.2f}+/-{r['std_a']:.2f}"
        b_str = f"{r['mean_b']:.2f}+/-{r['std_b']:.2f}"
        print(
            f"{name:<38} {a_str:>22} {b_str:>22} "
            f"{r['t_stat']:>9.3f} {r['p_value']:>10.4f} {sig:>4}"
        )


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_distribution(
    values,
    title: str,
    xlabel: str,
    save_path: str | None = None,
) -> None:
    """Histogram of a single metric's distribution with a mean marker."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins="auto", color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(
        arr.mean(),
        color="red",
        linestyle="--",
        label=f"mean = {arr.mean():.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    if save_path:
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    values_a,
    values_b,
    label_a: str,
    label_b: str,
    title: str,
    xlabel: str,
    save_path: str | None = None,
) -> None:
    """Overlaid histogram comparing two texts' distributions on one metric."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size == 0 and b.size == 0:
        return

    a_min = a.min() if a.size else (b.min() if b.size else 0.0)
    a_max = a.max() if a.size else (b.max() if b.size else 1.0)
    b_min = b.min() if b.size else a_min
    b_max = b.max() if b.size else a_max
    lo = min(a_min, b_min)
    hi = max(a_max, b_max)
    if hi == lo:
        hi = lo + 1
    bins = np.linspace(lo, hi, 21)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        a,
        bins=bins,
        alpha=0.55,
        label=f"{label_a} (mean={a.mean():.2f})",
        color="steelblue",
        edgecolor="black",
    )
    ax.hist(
        b,
        bins=bins,
        alpha=0.55,
        label=f"{label_b} (mean={b.mean():.2f})",
        color="darkorange",
        edgecolor="black",
    )
    ax.axvline(a.mean(), color="steelblue", linestyle="--")
    ax.axvline(b.mean(), color="darkorange", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    if save_path:
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_multi_comparison(
    values_list,
    labels,
    title: str,
    xlabel: str,
    save_path: str | None = None,
) -> None:
    """Overlaid histograms of multiple texts on a single metric."""
    arrays = [np.asarray(v, dtype=float) for v in values_list]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return
    all_vals = np.concatenate(arrays)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if hi == lo:
        hi = lo + 1
    bins = np.linspace(lo, hi, 21)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, (arr, label) in enumerate(zip(arrays, labels)):
        color = cmap(i % 10)
        ax.hist(
            arr,
            bins=bins,
            alpha=0.5,
            label=f"{label} (mean={arr.mean():.2f})",
            color=color,
            edgecolor="black",
        )
        ax.axvline(arr.mean(), color=color, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    if save_path:
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_all_distributions(
    metrics: dict[str, list],
    out_dir: str = "plots",
    prefix: str = "dist_",
) -> None:
    """Write a distribution histogram for every metric to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for name, values in metrics.items():
        if not values:
            continue
        safe = name.replace("/", "_").replace("\\", "_")
        plot_distribution(
            values,
            title=f"Distribution: {name}",
            xlabel=name,
            save_path=os.path.join(out_dir, f"{prefix}{safe}.png"),
        )


def _sentence_count(metrics: dict[str, list]) -> int | None:
    """Best-effort sentence count for a metric dict.

    Uses len(metrics["sentence_lengths"]) when available, otherwise the modal
    length across per-sentence arrays. Returns None if the dict is empty.
    """
    if "sentence_lengths" in metrics and metrics["sentence_lengths"]:
        return len(metrics["sentence_lengths"])
    lengths: dict[int, int] = {}
    for values in metrics.values():
        n = len(values)
        if n:
            lengths[n] = lengths.get(n, 0) + 1
    if not lengths:
        return None
    return max(lengths, key=lambda k: lengths[k])


def _normalize_metric_array(
    values_a, values_b, n_a: int, n_b: int
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Return aligned float arrays plus a skip reason, or (None, None, reason)."""
    a_empty = values_a is None or len(values_a) == 0
    b_empty = values_b is None or len(values_b) == 0
    if a_empty and b_empty:
        return None, None, "empty"
    if a_empty:
        return np.zeros(n_b, dtype=float), np.asarray(values_b, dtype=float), None
    if b_empty:
        return np.asarray(values_a, dtype=float), np.zeros(n_a, dtype=float), None
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return None, None, "n_lt_2"
    return a, b, None


def _combined_nonzero(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a) + np.count_nonzero(b))


def _capped_linear_weight(combined_nonzero: int, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    return float(min(1.0, combined_nonzero / threshold))


def _global_simple(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    min_appearances: int = 10,
) -> dict:
    """Combine per-metric KS distances using capped linear weights.

    Each metric contributes (1 - KS distance) scaled by
    min(1, combined_nonzero / min_appearances). The returned similarity is the
    weighted mean of those contributions; metrics with too few observations
    fall out before averaging, so dominant high-volume metrics do not skew the
    result while rare-feature noise does not dominate either.
    """
    n_a = _sentence_count(metrics_a) or 0
    n_b = _sentence_count(metrics_b) or 0
    all_names = sorted(set(metrics_a) | set(metrics_b))

    per_metric: dict[str, dict] = {}
    dropped: list[dict] = []
    numerator = 0.0
    denominator = 0.0

    for name in all_names:
        a, b, skip = _normalize_metric_array(
            metrics_a.get(name), metrics_b.get(name), n_a, n_b
        )
        if a is None:
            dropped.append({"name": name, "reason": skip})
            continue

        combined_col = np.concatenate([a, b])
        if np.std(combined_col) == 0:
            dropped.append({"name": name, "reason": "constant"})
            continue

        ks_stat, _ = ks_2samp(a, b)
        ks_stat = float(ks_stat)
        combined_nz = _combined_nonzero(a, b)
        weight = _capped_linear_weight(combined_nz, min_appearances)
        similarity = max(0.0, 1.0 - ks_stat)

        per_metric[name] = {
            "ks_distance": ks_stat,
            "combined_nonzero": combined_nz,
            "weight": weight,
            "similarity": similarity,
            "n_a": int(a.size),
            "n_b": int(b.size),
        }
        if weight > 0:
            numerator += weight * similarity
            denominator += weight

    similarity = numerator / denominator if denominator > 0 else 1.0
    used = sum(1 for v in per_metric.values() if v["weight"] > 0)
    return {
        "similarity": float(similarity),
        "method": "simple",
        "weighting": f"capped_linear,threshold={min_appearances}",
        "n_metrics": len(per_metric),
        "n_metrics_used": used,
        "n_metrics_dropped": len(dropped),
        "per_metric": per_metric,
        "dropped": dropped,
    }


def _drop_collinear_columns(
    cols: list[np.ndarray], names: list[str], threshold: float = 0.98
) -> tuple[list[np.ndarray], list[str], list[dict]]:
    """Greedily drop the last column of any near-duplicate pair."""
    kept_cols: list[np.ndarray] = list(cols)
    kept_names: list[str] = list(names)
    dropped: list[dict] = []
    for i in range(len(kept_cols) - 1, -1, -1):
        series_i = kept_cols[i]
        std_i = float(np.std(series_i))
        if std_i == 0:
            dropped.append({"name": kept_names[i], "reason": "zero_variance_in_run"})
            kept_cols.pop(i)
            kept_names.pop(i)
            continue
        redundant = False
        for j in range(i):
            series_j = kept_cols[j]
            std_j = float(np.std(series_j))
            if std_j == 0:
                continue
            corr = float(np.corrcoef(series_i, series_j)[0, 1])
            if np.isnan(corr) or abs(corr) >= threshold:
                redundant = True
                dropped.append({"name": kept_names[i], "reason": "collinear"})
                break
        if redundant:
            kept_cols.pop(i)
            kept_names.pop(i)
    return kept_cols, kept_names, dropped


def _select_per_sentence_features(
    metrics: dict[str, list], n_sentences: int
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Filter and align metric arrays to the canonical per-sentence size."""
    aligned: dict[str, np.ndarray] = {}
    dropped: list[dict] = []
    for name, values in metrics.items():
        arr = np.asarray(values, dtype=float) if values else np.zeros(0)
        if arr.size == 0:
            dropped.append({"name": name, "reason": "empty"})
            continue
        if arr.size != n_sentences:
            dropped.append({
                "name": name,
                "reason": "wrong_length",
                "size": int(arr.size),
                "expected": int(n_sentences),
            })
            continue
        aligned[name] = arr
    return aligned, dropped


def _global_manova(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    min_appearances: int = 10,
) -> dict:
    """Run a sentence-level MANOVA with Pillai's trace as the headline statistic.

    Features (per-sentence metrics) are zero-padded when missing in one text,
    filtered by combined non-zero count, then reduced by variance and absolute
    correlation thresholds before fitting standard statsmodels MANOVA. The
    returned similarity is 1 - Pillai's trace, where 1 means identical means
    across every retained feature.
    """
    n_a = _sentence_count(metrics_a)
    n_b = _sentence_count(metrics_b)
    if not n_a or not n_b:
        return {
            "similarity": None,
            "method": "manova",
            "error": "could_not_determine_sentence_count",
            "n_obs_a": n_a,
            "n_obs_b": n_b,
        }

    aligned_a, dropped_a = _select_per_sentence_features(metrics_a, n_a)
    aligned_b, dropped_b = _select_per_sentence_features(metrics_b, n_b)
    common = sorted(set(aligned_a) & set(aligned_b))

    rows_a, rows_b, dropped_common = [], [], []
    valid_names: list[str] = []
    for name in common:
        col_a = aligned_a[name]
        col_b = aligned_b[name]
        combined = np.concatenate([col_a, col_b])
        combined_nz = int(np.count_nonzero(combined))
        if combined_nz < min_appearances:
            dropped_common.append({
                "name": name,
                "reason": "sparse",
                "combined_nonzero": combined_nz,
            })
            continue
        if float(np.std(combined)) == 0:
            dropped_common.append({"name": name, "reason": "constant"})
            continue
        rows_a.append(col_a)
        rows_b.append(col_b)
        valid_names.append(name)

    if len(valid_names) < 2 or n_a + n_b - len(valid_names) - 1 <= 0:
        return {
            "similarity": None,
            "method": "manova",
            "error": "insufficient_features_or_samples",
            "n_features": len(valid_names),
            "n_obs_a": n_a,
            "n_obs_b": n_b,
            "dropped": dropped_a + dropped_b + dropped_common,
        }

    X = np.vstack([np.column_stack(rows_a), np.column_stack(rows_b)])
    kept_cols, kept_names, dropped_corr = _drop_collinear_columns(
        [X[:, i] for i in range(X.shape[1])], valid_names
    )

    if len(kept_names) < 2 or n_a + n_b - len(kept_names) - 1 <= 0:
        return {
            "similarity": None,
            "method": "manova",
            "error": "insufficient_features_after_collinearity",
            "n_features": len(kept_names),
            "n_obs_a": n_a,
            "n_obs_b": n_b,
            "dropped": dropped_a + dropped_b + dropped_common + dropped_corr,
        }

    X_final = np.column_stack(kept_cols)
    df = pd.DataFrame(X_final, columns=kept_names)
    df["group"] = ["A"] * n_a + ["B"] * n_b
    formula = " + ".join(kept_names) + " ~ group"

    try:
        result = MANOVA.from_formula(formula, data=df).mv_test()
        stat_df = result.results["group"]["stat"]
        pillai = float(stat_df.loc["Pillai's trace", "Value"])
        wilks = float(stat_df.loc["Wilks' lambda", "Value"])
        hotelling = float(stat_df.loc["Hotelling-Lawley trace", "Value"])
        roy = float(stat_df.loc["Roy's greatest root", "Value"])
        f_value = float(stat_df.loc["Pillai's trace", "F Value"])
        p_value = float(stat_df.loc["Pillai's trace", "Pr > F"])
        num_df = int(stat_df.loc["Pillai's trace", "Num DF"])
        den_df = float(stat_df.loc["Pillai's trace", "Den DF"])
    except (KeyError, ValueError) as exc:
        return {
            "similarity": None,
            "method": "manova",
            "error": f"manova_failed:{exc}",
            "features_used": kept_names,
            "n_obs_a": n_a,
            "n_obs_b": n_b,
            "dropped": dropped_a + dropped_b + dropped_common + dropped_corr,
        }

    return {
        "similarity": float(max(0.0, min(1.0, 1.0 - pillai))),
        "method": "manova",
        "n_features": len(kept_names),
        "n_obs_a": n_a,
        "n_obs_b": n_b,
        "features_used": kept_names,
        "pillai_trace": pillai,
        "wilks_lambda": wilks,
        "hotelling_lawley_trace": hotelling,
        "roy_greatest_root": roy,
        "f_value": f_value,
        "p_value": p_value,
        "numerator_df": num_df,
        "denominator_df": den_df,
        "dropped": dropped_a + dropped_b + dropped_common + dropped_corr,
    }


def global_similarity(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    method: str = "simple",
    min_appearances: int = 10,
) -> dict:
    """Compute a single [0, 1] similarity score between two metric dicts.

    Parameters
    ----------
    metrics_a, metrics_b : dict[str, list]
        Output of get_all_metrics for each text.
    method : {"simple", "manova"}
        "simple" averages per-metric Kolmogorov-Smirnov distances using
        capped linear weights derived from combined non-zero observations.
        "manova" runs a sentence-level statsmodels MANOVA over the
        per-sentence feature matrix and reports 1 - Pillai's trace.
    min_appearances : int
        Threshold for the capped linear weight (simple) and the absolute
        sparsity filter on combined non-zero observations (MANOVA).

    Returns
    -------
    dict
        result["similarity"] is the headline 0-1 score; the result dict also
        carries method, weighting details, per-metric stats, and a list of
        dropped metrics with reasons.
    """
    if method == "simple":
        return _global_simple(metrics_a, metrics_b, min_appearances)
    if method == "manova":
        return _global_manova(metrics_a, metrics_b, min_appearances)
    raise ValueError(f"unknown method: {method!r}; expected 'simple' or 'manova'")


def print_global_similarity(
    result: dict,
    label_a: str = "A",
    label_b: str = "B",
    top: int = 5,
) -> None:
    """Pretty-print a global_similarity result and its strongest contributors."""
    sim = result.get("similarity")
    sim_str = f"{sim:.4f}" if sim is not None else "N/A"
    print(
        f"Global similarity ({result.get('method')}): {sim_str}  "
        f"[{label_a} vs {label_b}]"
    )
    if result.get("method") == "manova":
        print(
            f"  Pillai trace: {result['pillai_trace']:.4f}  "
            f"F({result['numerator_df']}, {result['denominator_df']:.1f}) "
            f"= {result['f_value']:.3f}  p = {result['p_value']:.4g}"
        )
        print(
            f"  features used: {result['n_features']}  "
            f"obs: A={result['n_obs_a']} B={result['n_obs_b']}"
        )
    else:
        print(
            f"  metrics used: {result['n_metrics_used']}  "
            f"dropped: {result['n_metrics_dropped']}  "
            f"weighting: {result['weighting']}"
        )

    per = result.get("per_metric")
    if per and top > 0:
        ranked = sorted(
            per.items(),
            key=lambda kv: kv[1]["weight"] * (1.0 - kv[1]["ks_distance"]),
            reverse=True,
        )
        print(f"  top {top} contributors (weight x similarity):")
        for name, info in ranked[:top]:
            print(
                f"    {name:<38} w={info['weight']:.2f} "
                f"ks={info['ks_distance']:.3f} sim={info['similarity']:.3f}"
            )

    dropped = result.get("dropped")
    if dropped:
        by_reason: dict[str, int] = {}
        for d in dropped:
            by_reason[d.get("reason", "?")] = by_reason.get(d.get("reason", "?"), 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
        print(f"  dropped summary: {summary}")

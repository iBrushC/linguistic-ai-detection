# Analysis of stylometry on general works of text

import os
import json
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


_SPIDER_GROUPS = {
    "connective_": "Connective Density by Type",
    "dep_": "Syntactic Dependency Distribution",
    "pos_": "Part-of-Speech Distribution",
}


def _spider_axis_mean(values) -> float:
    arr = np.asarray(values, dtype=float) if values else np.zeros(0)
    return float(arr.mean()) if arr.size else 0.0


def _spider_min_max(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return [0.5 for _ in values]
    return ((arr - lo) / (hi - lo)).tolist()


def plot_spider_charts(
    metrics_list: list[dict[str, list]],
    labels: list[str],
    out_dir: str = "plots",
    prefix: str = "spider_",
) -> None:
    """Write one radar chart per metric group (connective_, dep_, pos_).

    Each chart overlays one polygon per input text sharing the same axes
    (metric names within the group). Per-axis values are the mean per-sentence
    counts across the text, then min-max normalized to [0, 1] across all texts
    so different scales line up on the same plot. Axes whose value is constant
    across all texts (e.g. a tag that never appears) are dropped. The plot's
    POS group uses the bare tag (NN, VB, ...) as the axis label rather than
    the full description so the chart stays readable.
    """
    if len(metrics_list) != len(labels):
        raise ValueError("metrics_list and labels must have the same length")
    if not metrics_list:
        return

    os.makedirs(out_dir, exist_ok=True)

    for group_prefix, base_title in _SPIDER_GROUPS.items():
        all_names: list[str] = []
        seen: set[str] = set()
        for m in metrics_list:
            for k in m:
                if k.startswith(group_prefix) and k not in seen:
                    seen.add(k)
                    all_names.append(k)
        all_names.sort()
        if len(all_names) < 3:
            continue

        raw_columns: list[list[float]] = []
        axis_labels: list[str] = []
        for name in all_names:
            means = [_spider_axis_mean(m.get(name)) for m in metrics_list]
            if float(np.std(means)) == 0.0:
                continue
            raw_columns.append(means)
            axis_labels.append(name[len(group_prefix):])

        if len(axis_labels) < 3:
            continue

        per_text = [list(row) for row in zip(*raw_columns)]
        normalized = [_spider_min_max(row) for row in per_text]

        n_axes = len(axis_labels)
        angles = [i / float(n_axes) * 2.0 * np.pi for i in range(n_axes)]
        angles_closed = angles + angles[:1]

        fig_size = max(8.0, 1.2 + 0.45 * n_axes)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw=dict(polar=True))
        cmap = plt.get_cmap("tab10")
        for i, label in enumerate(labels):
            values = normalized[i] + normalized[i][:1]
            color = cmap(i % 10)
            ax.plot(angles_closed, values, color=color, linewidth=2, label=label)
            ax.fill(angles_closed, values, color=color, alpha=0.15)

        ax.set_xticks(angles)
        ax.set_xticklabels(axis_labels, fontsize=9)
        ax.tick_params(axis="x", pad=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
        ax.set_title(base_title, pad=28, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08))
        fig.tight_layout()

        filename = f"{prefix}{group_prefix.rstrip('_')}.png"
        save_path = os.path.join(out_dir, filename)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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


def _all_feature_stats(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
) -> pd.DataFrame:
    """Build a per-metric summary table (means, stds, sample sizes) for two texts.

    Used by plot_all_features_overlap. Rows for metrics that are empty in both
    texts are dropped; metrics present in only one text are kept with zeros on
    the missing side so the chart still shows a comparison.
    """
    rows: list[dict] = []
    for name in sorted(set(metrics_a) | set(metrics_b)):
        a_vals = np.asarray(metrics_a.get(name) or [], dtype=float)
        b_vals = np.asarray(metrics_b.get(name) or [], dtype=float)
        if a_vals.size == 0 and b_vals.size == 0:
            continue
        mean_a, std_a, n_a = _metric_summary(a_vals) if a_vals.size else (0.0, 0.0, 0)
        mean_b, std_b, n_b = _metric_summary(b_vals) if b_vals.size else (0.0, 0.0, 0)
        rows.append({
            "name": name,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "n_a": n_a,
            "n_b": n_b,
        })
    return pd.DataFrame(rows)


def plot_all_features_overlap(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    label_a: str,
    label_b: str,
    out_dir: str = "plots",
    prefix: str = "all_features_",
) -> pd.DataFrame:
    """Write two complementary charts comparing every metric for two texts.

    Chart 1 (``prefix}distributions.png``): small-multiples grid, one subplot
    per metric. Each subplot overlays the two authors' per-sentence/per-word
    distributions as semi-transparent histograms on a shared bin range, so
    every metric gets its own axis. This sidesteps the scale problem of a
    single shared axis (sentence lengths ~120, TTR ~0.7, cleft counts ~0
    cannot all render at a useful size on one y-axis) and keeps both
    distributions fully visible at every metric.

    Chart 2 (``prefix}divergence.png``): horizontal bar chart of per-metric
    divergence, sorted largest first. Divergence is |mean_a - mean_b| /
    pooled-std (a.k.a. Cohen's d), which puts every metric on a common
    standardized scale. Metrics whose pooled std is 0 are plotted as 0 with a
    warning printed; they cannot separate the texts on their own.

    Together these charts answer the question "do some metrics dominate and
    hide real differences between authors?" Chart 1 exposes the raw shape
    of each per-author distribution; Chart 2 exposes standardized
    separation so metrics with very different natural magnitudes can be
    compared on a common axis.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = _all_feature_stats(metrics_a, metrics_b)
    if df.empty:
        return df

    metric_names = df["name"].tolist()
    n = len(metric_names)

    # ---- Chart 1: small multiples of per-metric distributions ----
    ncols = 5 if n > 5 else max(n, 1)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.4 * nrows + 0.6),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, name in enumerate(metric_names):
        ax = axes_flat[i]
        a_vals = np.asarray(metrics_a.get(name) or [], dtype=float)
        b_vals = np.asarray(metrics_b.get(name) or [], dtype=float)

        if a_vals.size == 0 and b_vals.size == 0:
            ax.set_visible(False)
            continue

        if a_vals.size and b_vals.size:
            combined = np.concatenate([a_vals, b_vals])
        elif a_vals.size:
            combined = a_vals
        else:
            combined = b_vals

        lo, hi = float(combined.min()), float(combined.max())
        if hi == lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 12)

        if a_vals.size:
            ax.hist(
                a_vals,
                bins=bins,
                density=True,
                alpha=0.55,
                label=label_a,
                color="steelblue",
                edgecolor="black",
            )
        if b_vals.size:
            ax.hist(
                b_vals,
                bins=bins,
                density=True,
                alpha=0.55,
                label=label_b,
                color="darkorange",
                edgecolor="black",
            )

        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"All Features Comparison: {label_a} vs {label_b}",
        fontsize=12,
        y=0.995,
    )
    first_ax = next((ax for ax in axes_flat if ax.get_visible()), None)
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=2,
                bbox_to_anchor=(0.5, 0.005),
                fontsize=10,
                frameon=True,
            )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(
        os.path.join(out_dir, f"{prefix}distributions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Chart 2: per-metric standardized divergence ----
    pooled_sq = (df["std_a"].fillna(0.0) ** 2 + df["std_b"].fillna(0.0) ** 2) / 2.0
    pooled = np.sqrt(pooled_sq)
    abs_diff = (df["mean_a"] - df["mean_b"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        cohens_d = np.where(pooled > 0, abs_diff / pooled, np.nan)
    df = df.copy()
    df["cohens_d"] = cohens_d

    zero_var = df[df["cohens_d"].isna()]["name"].tolist()
    if zero_var:
        print(
            f"[all_features_overlap] pooled std = 0 (no divergence possible) "
            f"for: {zero_var}"
        )
    df["cohens_d"] = df["cohens_d"].fillna(0.0)
    df_sorted = df.sort_values("cohens_d", ascending=True)  # largest at top of horizontal chart

    n_metrics = len(df_sorted)
    fig_h = max(6.0, 0.32 * n_metrics + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))
    y = np.arange(n_metrics)
    ax.barh(
        y,
        df_sorted["cohens_d"].values,
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["name"].tolist(), fontsize=8)
    ax.set_xlabel("|mean_a - mean_b| / pooled std  (Cohen's d, higher = more separating)")
    ax.set_title(
        f"Per-Metric Divergence: {label_a} vs {label_b}\n"
        "(standardized mean difference; metrics with the largest d are the "
        "strongest separators between these two authors)"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"{prefix}divergence.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return df


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
        return np.zeros(n_a, dtype=float), np.asarray(values_b, dtype=float), None
    if b_empty:
        return np.asarray(values_a, dtype=float), np.zeros(n_b, dtype=float), None
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
    metric_weights: dict[str, float] | None = None,
) -> dict:
    """Combine per-metric KS distances using capped linear weights.

    Each metric contributes (1 - KS distance) scaled by
    min(1, combined_nonzero / min_appearances). The returned similarity is the
    weighted mean of those contributions; metrics with too few observations
    fall out before averaging, so dominant high-volume metrics do not skew the
    result while rare-feature noise does not dominate either.

    When ``metric_weights`` is provided, each metric's capped linear weight is
    additionally scaled by ``metric_weights[name]``. The scaled weight is also
    reported in ``per_metric[name]["tuned_weight"]`` so callers can audit which
    features drove the final similarity.
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
        base_weight = _capped_linear_weight(combined_nz, min_appearances)
        weight = _apply_metric_weights(base_weight, name, metric_weights)
        similarity = max(0.0, 1.0 - ks_stat)

        per_metric[name] = {
            "ks_distance": ks_stat,
            "combined_nonzero": combined_nz,
            "weight": weight,
            "base_weight": base_weight,
            "similarity": similarity,
            "n_a": int(a.size),
            "n_b": int(b.size),
        }
        if weight > 0:
            numerator += weight * similarity
            denominator += weight

    similarity = numerator / denominator if denominator > 0 else 1.0
    used = sum(1 for v in per_metric.values() if v["weight"] > 0)
    weighting_desc = f"capped_linear,threshold={min_appearances}"
    if metric_weights:
        weighting_desc += ",tuned"
    return {
        "similarity": float(similarity),
        "method": "simple",
        "weighting": weighting_desc,
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
    formula_names = [f"feature_{i}" for i in range(len(kept_names))]
    df = pd.DataFrame(X_final, columns=formula_names)
    df["group"] = ["A"] * n_a + ["B"] * n_b
    formula = " + ".join(formula_names) + " ~ group"

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


DEFAULT_METRIC_WEIGHTS_FILENAME = "metric_weights.json"
METRIC_WEIGHTS_SCHEMA_VERSION = 3


def load_metric_weights(path: str | None) -> dict[str, float] | None:
    """Load a metric_weights.json file, returning None when unavailable.

    The file is expected to be produced by ``src.tune.compute_metric_weights``
    and contains a top-level ``metric_weights`` dict mapping metric name to a
    positive float. If ``path`` is None, the file does not exist, or it is
    malformed, None is returned and the caller treats weights as 1.0.
    """
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    weights = payload.get("metric_weights") if isinstance(payload, dict) else None
    if not isinstance(weights, dict):
        return None
    out: dict[str, float] = {}
    for name, value in weights.items():
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f > 0:
            out[str(name)] = f
    return out or None


def _apply_metric_weights(
    base_weight: float,
    name: str,
    metric_weights: dict[str, float] | None,
) -> float:
    if not metric_weights:
        return base_weight
    return float(base_weight * metric_weights.get(name, 1.0))


def global_similarity(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    method: str = "simple",
    min_appearances: int = 10,
    metric_weights: dict[str, float] | None = None,
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
        return _global_simple(metrics_a, metrics_b, min_appearances, metric_weights)
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

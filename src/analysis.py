# Analysis of stylometry on general works of text

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

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

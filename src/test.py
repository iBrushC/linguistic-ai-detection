# Main way of running the tests.
#
# CLI:
#   python test.py single                              # existing single-author smoke test
#   python test.py author-similarity                   # LOO self + concat cross matrix
#   python test.py author-similarity --method manova
#   python test.py author-similarity --out-dir DIR --workers N --no-cache
#
# Outputs from `author-similarity` are written to out-dir (default src/plots/):
#   - author_self_similarity.png      bar chart of LOO self-similarity per author
#   - author_similarity_heatmap.png   n x n heatmap of self + cross similarity
#   - author_similarity_matrix.json   full matrix + per-cell metadata
#   - .essay_metrics_cache.json       cached per-essay + concat-corpus metrics

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    compare_metrics,
    get_all_metrics,
    global_similarity,
    plot_distribution,
    plot_spider_charts,
    print_comparison,
    print_global_similarity,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
CACHE_FILENAME = ".essay_metrics_cache.json"
SELF_BAR_FILENAME = "author_self_similarity.png"
HEATMAP_FILENAME = "author_similarity_heatmap.png"
MATRIX_JSON_FILENAME = "author_similarity_matrix.json"


# ---------------------------------------------------------------------------
# IO + data prep helpers
# ---------------------------------------------------------------------------

def author_name(essay: dict) -> str:
    return essay["author"].replace("By ", "").strip()


def load_essays(path: str | None = None) -> list[dict]:
    if path is None:
        path = DEFAULT_ESSAYS_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_author(essays: list[dict]) -> dict[str, list[int]]:
    """Return {author_name: [essay_index, ...]} preserving source order."""
    grouped: dict[str, list[int]] = {}
    for i, e in enumerate(essays):
        grouped.setdefault(author_name(e), []).append(i)
    return grouped


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Metrics cache (per essay + per concatenated blob)
# ---------------------------------------------------------------------------

def _load_cache(out_dir: str) -> dict:
    path = os.path.join(out_dir, CACHE_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(out_dir: str, cache: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CACHE_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, path)


def _ensure_metrics(
    text: str,
    cache: dict,
    workers: int = 1,
) -> dict[str, list]:
    """Return get_all_metrics(text), using cache keyed by text fingerprint."""
    fp = _text_fingerprint(text)
    if fp in cache:
        return cache[fp]
    metrics = get_all_metrics(text)
    cache[fp] = metrics
    if workers <= 1:
        # Persist incrementally so an interrupted run keeps partial progress.
        try:
            _save_cache(os.path.dirname(_resolve_out_dir()), cache)
        except OSError:
            pass
    return metrics


def _resolve_out_dir() -> str:
    """Best-effort default cache directory; overwritten by --out-dir at runtime."""
    return DEFAULT_OUT_DIR


def _compute_essay_metrics_parallel(
    essays: list[dict],
    cache: dict,
    workers: int,
    label: str = "essay",
) -> None:
    """Populate cache for every essay that is not already cached."""
    pending: list[tuple[int, str]] = []
    for i, e in enumerate(essays):
        fp = _text_fingerprint(e["body"])
        if fp not in cache:
            pending.append((i, e["body"]))

    if not pending:
        print(f"[cache] all {label} metrics already cached")
        return

    print(f"[cache] computing metrics for {len(pending)} {label}(s) with {workers} worker(s)")

    def _job(body_text: str) -> tuple[str, dict[str, list]]:
        return _text_fingerprint(body_text), get_all_metrics(body_text)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_job, body) for _, body in pending]
            for fut in futures:
                fp, metrics = fut.result()
                cache[fp] = metrics
    else:
        for _, body in pending:
            fp, metrics = _job(body)
            cache[fp] = metrics


# ---------------------------------------------------------------------------
# Author-similarity core
# ---------------------------------------------------------------------------

def _concat_bodies(essays: list[dict], indices: list[int]) -> str:
    return "\n\n".join(essays[i]["body"] for i in indices)


def compute_self_similarity(
    author: str,
    indices: list[int],
    essays: list[dict],
    cache: dict,
    method: str,
) -> dict:
    """LOO self-similarity: average global_similarity over N permutations."""
    if len(indices) < 2:
        raise ValueError(
            f"author {author!r} has only {len(indices)} essay(s); need >=2 for LOO"
        )

    per_perm: list[float] = []
    for held_out in indices:
        corpus_idx = [i for i in indices if i != held_out]
        corpus_text = _concat_bodies(essays, corpus_idx)
        held_text = essays[held_out]["body"]

        corpus_metrics = _ensure_metrics(corpus_text, cache)
        held_metrics = _ensure_metrics(held_text, cache)

        result = global_similarity(corpus_metrics, held_metrics, method=method)
        sim = result.get("similarity")
        if sim is None:
            print(
                f"[self] {author} LOO held={held_out}: similarity is None "
                f"({result.get('error', 'no error key')}); treating as 0.0"
            )
            sim = 0.0
        per_perm.append(float(sim))
        print(f"[self] {author} LOO held={held_out}: {sim:.4f}")

    arr = np.asarray(per_perm, dtype=float)
    return {
        "author": author,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_permutations": int(arr.size),
        "per_permutation": [float(x) for x in per_perm],
    }


def compute_cross_similarity(
    author_a: str,
    indices_a: list[int],
    author_b: str,
    indices_b: list[int],
    essays: list[dict],
    cache: dict,
    method: str,
) -> float:
    """Concat(A) vs concat(B): single similarity score per author pair."""
    text_a = _concat_bodies(essays, indices_a)
    text_b = _concat_bodies(essays, indices_b)
    metrics_a = _ensure_metrics(text_a, cache)
    metrics_b = _ensure_metrics(text_b, cache)
    result = global_similarity(metrics_a, metrics_b, method=method)
    sim = result.get("similarity")
    if sim is None:
        print(
            f"[cross] {author_a} vs {author_b}: similarity is None "
            f"({result.get('error', 'no error key')}); treating as 0.0"
        )
        return 0.0
    print(f"[cross] {author_a} vs {author_b}: {sim:.4f}")
    return float(sim)


def build_similarity_matrix(
    grouped: dict[str, list[int]],
    essays: list[dict],
    cache: dict,
    method: str,
) -> pd.DataFrame:
    authors = sorted(grouped.keys())
    n = len(authors)
    matrix = pd.DataFrame(np.nan, index=authors, columns=authors)
    self_info: dict[str, dict] = {}

    print(f"[matrix] {n} authors: {authors}")

    # Diagonal: LOO self-similarity per author.
    for author in authors:
        indices = grouped[author]
        info = compute_self_similarity(author, indices, essays, cache, method)
        matrix.loc[author, author] = info["mean"]
        self_info[author] = info

    # Off-diagonal: concat-vs-concat, computed once per unordered pair then mirrored.
    for i, a in enumerate(authors):
        for j, b in enumerate(authors):
            if j <= i:
                continue
            sim = compute_cross_similarity(
                a, grouped[a], b, grouped[b], essays, cache, method
            )
            matrix.loc[a, b] = sim
            matrix.loc[b, a] = sim

    matrix.attrs["self_info"] = self_info
    matrix.attrs["method"] = method
    return matrix


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_self_similarity_bars(
    self_info: dict[str, dict],
    save_path: str,
) -> None:
    authors = list(self_info.keys())
    means = [self_info[a]["mean"] for a in authors]
    stds = [self_info[a]["std"] for a in authors]
    n_perms = [self_info[a]["n_permutations"] for a in authors]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    bars = ax.bar(
        range(len(authors)),
        means,
        yerr=stds,
        capsize=5,
        color=[cmap(i % 10) for i in range(len(authors))],
        edgecolor="black",
        alpha=0.85,
    )
    for bar, m, n in zip(bars, means, n_perms):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{m:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(authors)))
    ax.set_xticklabels(authors, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("LOO self-similarity (mean +/- std)")
    ax.set_title("Author Self-Similarity (Leave-One-Out)")
    ax.axhline(
        float(np.mean(means)),
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"grand mean = {np.mean(means):.3f}",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(
    matrix: pd.DataFrame,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    data = matrix.astype(float).values

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Author")
    ax.set_ylabel("Author")
    ax.set_title(
        "Author-Author Similarity Matrix\n"
        "(diagonal = LOO self-similarity, off-diagonal = concat vs concat)"
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "black" if 0.2 < val < 0.8 else "white"
            weight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                color=color, fontsize=9, fontweight=weight,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("similarity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subcommand: single (existing behavior, unchanged)
# ---------------------------------------------------------------------------

def cmd_single(_args: argparse.Namespace) -> None:
    essays = load_essays()
    first = author_name(essays[0])
    author_essays = [e for e in essays if author_name(e) == first]
    print(f"First author: {first} ({len(author_essays)} essays)")

    text_a = author_essays[0]["body"] + "\n" + author_essays[1]["body"]
    text_b = author_essays[2]["body"]
    label_a = f"{first} essays 1+2"
    label_b = f"{first} essay 3"

    metrics_a = get_all_metrics(text_a)
    metrics_b = get_all_metrics(text_b)

    print(f"\n=== {label_a} vs {label_b} ===\n")
    results = compare_metrics(metrics_a, metrics_b, label_a, label_b)
    print_comparison(results, label_a, label_b)

    print("\n--- global similarity (simple) ---")
    simple_result = global_similarity(metrics_a, metrics_b, method="simple")
    print_global_similarity(simple_result, label_a, label_b)

    print("\n--- global similarity (manova) ---")
    manova_result = global_similarity(metrics_a, metrics_b, method="manova")
    print_global_similarity(manova_result, label_a, label_b)

    out_dir = DEFAULT_OUT_DIR
    plot_path = os.path.join(out_dir, "words_per_sentence_distribution.png")
    plot_distribution(
        metrics_a["words_per_sentence"] + metrics_b["words_per_sentence"],
        title="Words per Sentence (all three essays combined)",
        xlabel="Words per Sentence",
        save_path=plot_path,
    )
    print(f"\nSaved words-per-sentence histogram to {plot_path}")

    print("\n--- generating spider charts ---")
    plot_spider_charts(
        [metrics_a, metrics_b],
        [label_a, label_b],
        out_dir=out_dir,
    )
    print(f"Saved spider charts to {out_dir}")


# ---------------------------------------------------------------------------
# Subcommand: author-similarity
# ---------------------------------------------------------------------------

def cmd_author_similarity(args: argparse.Namespace) -> None:
    essays = load_essays(args.essays)
    grouped = group_by_author(essays)
    if not grouped:
        raise SystemExit("no essays loaded")

    counts = {a: len(idx) for a, idx in grouped.items()}
    print(f"[init] loaded {len(essays)} essays across {len(grouped)} authors: {counts}")

    bad = [a for a, idx in grouped.items() if len(idx) < 2]
    if bad:
        raise SystemExit(
            f"authors with <2 essays cannot run LOO: {bad}; "
            "need at least 2 essays per author."
        )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    cache: dict = {} if args.no_cache else _load_cache(out_dir)

    def _persist():
        if not args.no_cache:
            _save_cache(out_dir, cache)

    # Override the default cache directory used inside _ensure_metrics.
    global _resolve_out_dir
    _resolve_out_dir = lambda: out_dir  # type: ignore[assignment]

    _compute_essay_metrics_parallel(essays, cache, args.workers, label="essay")
    _persist()

    matrix = build_similarity_matrix(grouped, essays, cache, args.method)
    self_info = matrix.attrs["self_info"]

    # Persist matrix JSON.
    matrix_path = os.path.join(out_dir, MATRIX_JSON_FILENAME)
    payload = {
        "method": args.method,
        "authors": list(matrix.index),
        "matrix": matrix.round(6).to_dict(orient="index"),
        "self_info": self_info,
    }
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[matrix] saved to {matrix_path}")

    # Plots.
    bar_path = os.path.join(out_dir, SELF_BAR_FILENAME)
    plot_self_similarity_bars(self_info, bar_path)
    print(f"[plot] saved {bar_path}")

    heat_path = os.path.join(out_dir, HEATMAP_FILENAME)
    plot_similarity_heatmap(matrix, heat_path)
    print(f"[plot] saved {heat_path}")

    # Console summary.
    print("\n=== Self-similarity (LOO) ===")
    header = f"{'author':<20} {'mean':>8} {'std':>8} {'n_perms':>8}  per_permutation"
    print(header)
    print("-" * len(header))
    for author in sorted(self_info):
        info = self_info[author]
        perms = ", ".join(f"{x:.3f}" for x in info["per_permutation"])
        print(
            f"{author:<20} {info['mean']:>8.4f} {info['std']:>8.4f} "
            f"{info['n_permutations']:>8d}  [{perms}]"
        )

    print("\n=== Similarity matrix ===")
    authors_list = list(matrix.index)
    name_w = max(len("author"), max(len(a) for a in authors_list))
    cell_w = max(8, max(len(a) for a in authors_list)) + 2
    header = " " * (name_w + 2) + "".join(
        f"{a:>{cell_w}}" for a in authors_list
    )
    print(header)
    print("-" * len(header))
    for a in authors_list:
        row_vals = "".join(
            f"{matrix.loc[a, b]:>{cell_w}.4f}" for b in authors_list
        )
        print(f"{a:<{name_w}}  {row_vals}")

    # Quick validation summary.
    diag = np.diag(matrix.astype(float).values)
    mask = ~np.eye(len(matrix), dtype=bool)
    off = matrix.astype(float).values[mask]
    print(
        f"\n[validation] diagonal mean = {diag.mean():.4f}  "
        f"off-diagonal mean = {off.mean():.4f}  "
        f"gap = {diag.mean() - off.mean():+.4f}"
    )

    _persist()


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stylometry tests for the linguistic-AI-detection project."
    )
    sub = parser.add_subparsers(dest="command")

    p_single = sub.add_parser(
        "single",
        help="Run the original single-author smoke test (unchanged).",
    )
    p_single.set_defaults(func=cmd_single)

    p_sim = sub.add_parser(
        "author-similarity",
        help="LOO self-similarity + concat-vs-concat cross similarity matrix.",
    )
    p_sim.add_argument(
        "--method",
        choices=["simple", "manova"],
        default="simple",
        help="global_similarity method (default: simple).",
    )
    p_sim.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Where to write plots + matrix JSON (default: {DEFAULT_OUT_DIR}).",
    )
    p_sim.add_argument(
        "--essays",
        default=None,
        help="Path to essays.json (default: repo root essays.json).",
    )
    p_sim.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent workers for get_all_metrics (default: 1).",
    )
    p_sim.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and overwrite the per-essay metrics cache.",
    )
    p_sim.set_defaults(func=cmd_author_similarity)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

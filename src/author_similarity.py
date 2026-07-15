"""5x5 author-level self/cross similarity heatmap.

Given the per-chunk distance matrices already produced by
``src.run_pipeline``, aggregate them into an ``n_authors x n_authors``
matrix where

    cell[i, j] = mean(1 - distance) over chunk pairs (a, b)
                 with author(a) = i, author(b) = j

Same-source-document pairs are dropped to keep the cross-author cells
from being inflated by shared topic. The diagonal averages over the
same-author, different-source-document chunk pairs (i.e. the same
population that drives the ``is_self`` positive class in
``src.evaluation``); the off-diagonal averages over cross-author pairs.

By default only ``cosine_delta`` is rendered (it is the most
discriminating of the three metrics, see ``run_summary.json``). The
displayed similarity values are min-max normalised across the full
matrix so the lowest cell is 0 and the highest is 1, regardless of the
underlying distance scale. Raw 5x5 matrices are also written as JSON.

CLI::

    python -m src.author_similarity
    python -m src.author_similarity --metric burrows_delta
    python -m src.author_similarity --distances-dir src/plots/distances \\
        --out-dir src/plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import chunking


METRICS = ("cosine_delta", "burrows_delta", "char_ngram_cosine")
DEFAULT_METRIC = "cosine_delta"


def _chunk_dict_to_text(chunks) -> list[str]:
    return [" ".join(c["tokens"]) for c in chunks]


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _sorted_unique_authors(authors: Sequence[str]) -> list[str]:
    """Author order: matches the corpus_report JSON (sorted alphabetically)."""
    return sorted(set(authors))


def author_similarity_matrix(
    distance_matrix: np.ndarray,
    authors: Sequence[str],
    source_doc_ids: Sequence[str],
    exclude_same_source_document: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build an n_authors x n_authors similarity matrix.

    Per author pair ``(i, j)``:

    1. average the per-chunk-pair distances over all qualifying pairs
       (same-author = self, different-author = cross; same-source-doc
       pairs dropped when ``exclude_same_source_document`` is true);
    2. when ``normalize=True``, scale distances by the off-diagonal
       range of ``distance_matrix`` so cosine-style metrics stay
       bounded, then invert: ``similarity = 1 - normalised_distance``;
    3. when ``normalize=True``, a final min-max pass stretches the
       similarity values across ``[0, 1]`` so the lowest cell is 0
       and the highest is 1, regardless of the underlying distance
       scale or how tightly the means are clustered.

    Returns ``(matrix, author_order)`` where ``matrix[i, j]`` is in
    ``[0, 1]`` and ``author_order[i]`` is the author label for
    row/column ``i``. The matrix is symmetric by construction.
    Diagonal entries use only same-author, different-source-document
    chunk pairs (the same population that drives the ``is_self``
    positive class in ``src.evaluation``).
    """
    authors_arr = np.asarray(authors)
    source_doc_ids_arr = np.asarray(source_doc_ids)
    author_order = _sorted_unique_authors(authors_arr)
    idx = {a: i for i, a in enumerate(author_order)}
    n = len(author_order)
    M_sum = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)

    n_chunks = len(authors_arr)
    iu, ju = np.triu_indices(n_chunks, k=1)
    if exclude_same_source_document:
        keep = source_doc_ids_arr[iu] != source_doc_ids_arr[ju]
        iu = iu[keep]
        ju = ju[keep]

    a_i = np.array([idx[a] for a in authors_arr[iu]], dtype=np.int64)
    a_j = np.array([idx[a] for a in authors_arr[ju]], dtype=np.int64)
    d = distance_matrix[iu, ju]

    for row, col, dist in zip(a_i, a_j, d):
        M_sum[row, col] += dist
        M_sum[col, row] += dist
        counts[row, col] += 1
        counts[col, row] += 1

    M = np.where(counts > 0, M_sum / np.maximum(counts, 1), np.nan)

    if normalize:
        finite_d = np.isfinite(distance_matrix)
        if finite_d.any():
            off_diag = distance_matrix.copy()
            np.fill_diagonal(off_diag, np.nan)
            finite_off = np.isfinite(off_diag)
            if finite_off.any():
                lo = float(off_diag[finite_off].min())
                hi = float(off_diag[finite_off].max())
            else:
                lo = float(distance_matrix[finite_d].min())
                hi = float(distance_matrix[finite_d].max())
            span = hi - lo
            if span > 0:
                M = np.where(np.isfinite(M), (M - lo) / span, np.nan)
            else:
                M = np.where(np.isfinite(M), 0.0, np.nan)

    M = 1.0 - M
    M = np.clip(M, 0.0, 1.0)

    if normalize:
        finite_M = np.isfinite(M)
        if finite_M.any():
            lo_M = float(M[finite_M].min())
            hi_M = float(M[finite_M].max())
            span_M = hi_M - lo_M
            if span_M > 0:
                M = np.where(np.isfinite(M), (M - lo_M) / span_M, np.nan)
            else:
                M = np.where(np.isfinite(M), 0.0, np.nan)

    return M, author_order


def plot_heatmap(
    matrix: np.ndarray,
    authors: Sequence[str],
    title: str,
    save_path: str,
    annotate: bool = True,
) -> None:
    """Write a single 5x5 heatmap to ``save_path``."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(authors)))
    ax.set_yticks(range(len(authors)))
    ax.set_xticklabels(authors, rotation=35, ha="right")
    ax.set_yticklabels(authors)
    ax.set_xlabel("author")
    ax.set_ylabel("author")
    ax.set_title(title)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean similarity = 1 - distance")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    matrices: dict[str, np.ndarray],
    authors: Sequence[str],
    save_path: str,
) -> None:
    """Side-by-side 5x5 heatmaps, one per metric, sharing a colour scale."""
    metrics = list(matrices.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    vmin, vmax = 0.0, 1.0
    for ax, metric in zip(axes, metrics):
        M = matrices[metric]
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(authors)))
        ax.set_yticks(range(len(authors)))
        ax.set_xticklabels(authors, rotation=35, ha="right")
        if ax is axes[0]:
            ax.set_yticklabels(authors)
        else:
            ax.set_yticklabels(authors)
        ax.set_title(metric)
        ax.set_xlabel("author")
        if ax is axes[0]:
            ax.set_ylabel("author")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Author self/cross similarity (diagonal = self, off-diag = cross)",
                 y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build(
    config: dict,
    distances_dir: str,
    out_dir: str,
    metrics: Sequence[str] = (DEFAULT_METRIC,),
) -> dict:
    """Compute and write the 5x5 similarity matrices. Returns a summary dict."""
    essays_path = os.path.join(REPO_ROOT, config["essays_path"])
    print(f"[author-similarity] loading essays from {essays_path}")
    with open(essays_path, encoding="utf-8") as f:
        test_articles = json.load(f)

    print(f"[author-similarity] chunking (chunk_size={config['chunk_size_tokens']}, "
          f"min_fill={config['min_chunk_fill_ratio']})")
    chunks = chunking.chunk_corpus(
        test_articles,
        chunk_size=int(config["chunk_size_tokens"]),
        min_fill_ratio=float(config["min_chunk_fill_ratio"]),
    )
    chunking.enforce_minimum(chunks, int(config["min_chunks_per_author"]))

    authors = [c["author"] for c in chunks]
    source_doc_ids = [c["source_doc_id"] for c in chunks]
    author_order = _sorted_unique_authors(authors)
    print(f"[author-similarity] {len(chunks)} chunks across {len(author_order)} authors: "
          f"{author_order}")

    os.makedirs(out_dir, exist_ok=True)

    matrices: dict[str, np.ndarray] = {}
    payloads: dict[str, dict] = {}
    for metric in metrics:
        dist_path = os.path.join(distances_dir, f"{metric}.npy")
        if not os.path.isfile(dist_path):
            print(f"[author-similarity] skipping {metric}: {dist_path} not found")
            continue
        D = np.load(dist_path)
        if D.shape[0] != len(chunks):
            raise SystemExit(
                f"{metric}: distance matrix has {D.shape[0]} rows but "
                f"chunking produced {len(chunks)} chunks; "
                f"re-run the pipeline with the same chunk_size."
            )
        M, _ = author_similarity_matrix(
            D, authors, source_doc_ids,
            exclude_same_source_document=bool(config["exclude_same_source_document"]),
        )
        matrices[metric] = M

        out_png = os.path.join(out_dir, f"author_similarity_{metric}.png")
        plot_heatmap(
            M, author_order,
            title=f"{metric}: 5x5 author self/cross similarity",
            save_path=out_png,
        )
        print(f"[author-similarity] wrote {out_png}")

        payload = {
            "metric": metric,
            "authors": author_order,
            "matrix": [[float(v) if not np.isnan(v) else None for v in row] for row in M],
            "exclude_same_source_document": bool(config["exclude_same_source_document"]),
            "n_chunks": len(chunks),
        }
        payloads[metric] = payload
        _save_json_atomic(
            os.path.join(out_dir, f"author_similarity_{metric}.json"),
            payload,
        )

    if len(matrices) > 1:
        combined_path = os.path.join(out_dir, "author_similarity_combined.png")
        plot_combined(matrices, author_order, combined_path)
        print(f"[author-similarity] wrote {combined_path}")

    summary = {
        "authors": author_order,
        "n_chunks": len(chunks),
        "exclude_same_source_document": bool(config["exclude_same_source_document"]),
        "per_metric": payloads,
    }
    _save_json_atomic(os.path.join(out_dir, "author_similarity_summary.json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config",
                   default=os.path.join(REPO_ROOT, "src", "configs", "default.json"))
    p.add_argument("--distances-dir",
                   default=os.path.join(REPO_ROOT, "src", "plots", "distances"))
    p.add_argument("--out-dir",
                   default=os.path.join(REPO_ROOT, "src", "plots"))
    p.add_argument("--metric", action="append", choices=METRICS, default=None,
                   help="Restrict to one metric (repeatable). Default: cosine_delta.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    metrics = list(args.metric) if args.metric else [DEFAULT_METRIC]
    build(
        cfg,
        distances_dir=args.distances_dir,
        out_dir=args.out_dir,
        metrics=metrics,
    )
    return 0


__all__ = [
    "METRICS",
    "DEFAULT_METRIC",
    "author_similarity_matrix",
    "plot_heatmap",
    "plot_combined",
    "build",
]


if __name__ == "__main__":
    raise SystemExit(main())
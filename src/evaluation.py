"""Evaluation utilities: AUC of self-vs-cross distance distributions,
permutation p-value, and overlap plots.

Pairwise distance samples are NOT independent observations (a single
chunk contributes to N-1 pairs), so ordinary significance tests on the
scalar distances are invalid. The permutation test shuffles author
labels while preserving the per-author chunk counts, recomputes the AUC
on the shuffled assignment, and reports the empirical p-value
(fraction of permuted AUCs ≥ observed AUC).
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def _build_pair_labels(
    authors: Sequence[str],
    source_doc_ids: Sequence[str],
    exclude_same_source_document: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (``upper_tri_distances``, ``is_self``, ``author_pair_index``).

    ``is_self`` is 1 when the two chunks are by the same author and 0
    otherwise. Pairs of chunks sharing a source document are dropped
    when ``exclude_same_source_document`` is True (default).

    Returns flat arrays for the strict upper triangle only, so every
    pair appears exactly once.
    """
    n = len(authors)
    authors = np.asarray(authors)
    source_doc_ids = np.asarray(source_doc_ids)
    iu, ju = np.triu_indices(n, k=1)
    if exclude_same_source_document:
        keep = source_doc_ids[iu] != source_doc_ids[ju]
        iu = iu[keep]
        ju = ju[keep]
    is_self = (authors[iu] == authors[ju]).astype(np.int32)
    return iu, ju, is_self


def pair_labels(
    dist_matrix: np.ndarray,
    authors: Sequence[str],
    source_doc_ids: Sequence[str],
    exclude_same_source_document: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distance_values, is_self_array)`` over kept pairs.

    Lower triangle is dropped; diagonal is dropped. When
    ``exclude_same_source_document`` is true, pairs of chunks from the
    same source article are also dropped.
    """
    iu, ju, is_self = _build_pair_labels(
        authors, source_doc_ids, exclude_same_source_document
    )
    return dist_matrix[iu, ju], is_self


def auc_self_vs_cross(
    distances: np.ndarray,
    is_self: np.ndarray,
) -> float:
    """AUC where the positive class is *self* (same author).

    Distance is a *dissimilarity*, so we negate so higher score = more
    likely to be the same author.
    """
    if is_self.sum() == 0 or is_self.sum() == len(is_self):
        return float("nan")
    return float(roc_auc_score(is_self, -distances))


def overlap_plot(
    distances: np.ndarray,
    is_self: np.ndarray,
    metric_name: str,
    save_path: str,
    auc: float | None = None,
    perm_pvalue: float | None = None,
) -> None:
    """Two histograms (self vs cross) of the distance values."""
    self_d = distances[is_self == 1]
    cross_d = distances[is_self == 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    if self_d.size:
        ax.hist(self_d, bins=30, alpha=0.55, label=f"self (n={self_d.size})", color="steelblue", edgecolor="black")
    if cross_d.size:
        ax.hist(cross_d, bins=30, alpha=0.55, label=f"cross (n={cross_d.size})", color="darkorange", edgecolor="black")
    title_extra = ""
    if auc is not None:
        title_extra += f"   AUC = {auc:.4f}"
    if perm_pvalue is not None:
        title_extra += f"   p(perm) = {perm_pvalue:.4g}"
    ax.set_title(f"{metric_name}: pairwise distance distribution{title_extra}")
    ax.set_xlabel("distance")
    ax.set_ylabel("pair count")
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def permutation_auc(
    authors: Sequence[str],
    source_doc_ids: Sequence[str] | None = None,
    distance_matrix: np.ndarray | None = None,
    distance_fn=None,
    is_self: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    iu: np.ndarray | None = None,
    ju: np.ndarray | None = None,
    n_permutations: int = 250,
    rng_seed: int | None = 1729,
    exclude_same_source_document: bool = True,
) -> dict:
    """Empirical p-value for the observed AUC vs label-shuffled AUCs.

    Pass either the precomputed pair distances + labels ``(distances,
    is_self)`` (fast path: avoids recomputing the distance matrix) or
    a callable ``distance_fn(authors_perm, iu, ju) -> (distances,
    is_self)`` that recomputes under a permutation.

    Label shuffles preserve per-author chunk counts: each chunk keeps
    its *position* but its author label is reassigned without
    replacement. The permutation null respects the original class
    balance.
    """
    rng = np.random.default_rng(rng_seed)
    authors_arr = np.asarray(authors)

    if iu is None or ju is None:
        if source_doc_ids is None or distances is None:
            raise ValueError(
                "either pass (iu, ju) explicitly or supply source_doc_ids + distances"
            )
        iu, ju, _ = _build_pair_labels(
            authors, source_doc_ids, exclude_same_source_document
        )
        iu = np.asarray(iu)
        ju = np.asarray(ju)

    if distances is None or is_self is None:
        if distance_fn is None:
            raise ValueError("provide either distances+is_self or distance_fn")
        distances, is_self = distance_fn(authors_arr, iu, ju)
    observed_auc = auc_self_vs_cross(distances, is_self)

    permuted_aucs = np.empty(n_permutations, dtype=np.float64)
    for k in range(n_permutations):
        perm = authors_arr.copy()
        perm[:] = rng.permutation(perm)
        if distance_fn is not None:
            d_perm, lbl_perm = distance_fn(perm, iu, ju)
        else:
            d_perm = distances
            lbl_perm = (perm[iu] == perm[ju]).astype(np.int32)
        if lbl_perm.sum() == 0 or lbl_perm.sum() == len(lbl_perm):
            # a permutation that produces only one class cannot yield an AUC;
            # record NaN and skip from the empirical-distribution sample
            permuted_aucs[k] = float("nan")
            continue
        permuted_aucs[k] = auc_self_vs_cross(d_perm, lbl_perm)

    valid = np.isfinite(permuted_aucs)
    n_valid = int(valid.sum())
    if n_valid == 0:
        pvalue = float("nan")
    else:
        ge = int((permuted_aucs[valid] >= observed_auc).sum())
        pvalue = (1 + ge) / (1 + n_valid)
    return {
        "observed_auc": float(observed_auc),
        "n_permutations": int(n_permutations),
        "n_valid_permutations": n_valid,
        "p_value": float(pvalue),
        "permuted_auc_mean": float(np.nanmean(permuted_aucs)) if n_valid else float("nan"),
        "permuted_auc_std": float(np.nanstd(permuted_aucs, ddof=1)) if n_valid > 1 else 0.0,
        "permuted_auc_50": float(np.nanmedian(permuted_aucs)) if n_valid else float("nan"),
        "permuted_auc_95": float(np.nanquantile(permuted_aucs, 0.95)) if n_valid else float("nan"),
        "permuted_auc_99": float(np.nanquantile(permuted_aucs, 0.99)) if n_valid else float("nan"),
        "observed_ge_count": int((permuted_aucs[valid] >= observed_auc).sum()) if n_valid else 0,
    }


__all__ = [
    "pair_labels",
    "auc_self_vs_cross",
    "overlap_plot",
    "permutation_auc",
]

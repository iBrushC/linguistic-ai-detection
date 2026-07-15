# Deterministic per-metric weighting derived from Cohen's d on essay pairs.
#
# Approach:
#   1. Load every essay from essays.json and group by author.
#   2. Enumerate all unordered essay pairs (i, j); tag each as same-author or
#      different-author.
#   3. For each pair, compute Cohen's d per metric over the per-sentence
#      distributions (pooled std, mirroring analysis.py's convention).
#   4. Aggregate per metric: mean d across same-author pairs (d_S) and across
#      different-author pairs (d_D).
#   5. Separation score: sep = (d_D + eps) / (d_S + eps). Higher sep means a
#      metric discriminates authors more than it varies within one author.
#   6. Convert to a relative weight centered on 1.0 over metrics that
#      separated authors (sep > 1). Anti-signal metrics clamp to 1.0 so they
#      neither help nor hurt the simple similarity aggregator.
#   7. Persist to <out_dir>/metric_weights.json plus a debug sidecar.
#
# CLI:
#   python tune.py                                # default plots dir, regex
#   python tune.py --out-dir DIR --essays PATH    # custom output / corpus
#   python tune.py --no-debug                     # skip debug sidecar

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    DEFAULT_METRIC_WEIGHTS_FILENAME,
    METRIC_WEIGHTS_SCHEMA_VERSION,
    get_all_metrics,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
DEBUG_FILENAME = "metric_weights_debug.json"
EPS = 1e-6
CLAMP_MIN = 0.1


def author_name(essay: dict) -> str:
    return essay["author"].replace("By ", "").strip()


def group_by_author(essays: list[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for i, e in enumerate(essays):
        grouped.setdefault(author_name(e), []).append(i)
    return grouped


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pooled Cohen's d, mirroring analysis.py's all_features_overlap panel.

    Returns None when the metric cannot be evaluated (too few samples or zero
    pooled variance).
    """
    if a.size < 2 or b.size < 2:
        return None
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_var = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    if pooled_var <= 0:
        return None
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return None
    return abs(float(a.mean()) - float(b.mean())) / pooled_std


def _pair_metric_array(metrics: dict[str, list], name: str) -> np.ndarray | None:
    values = metrics.get(name)
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    return arr


def compute_metric_weights(
    essays: list[dict],
    *,
    metrics_lookup: dict[int, dict[str, list]],
) -> dict:
    """Compute per-metric weights from pairwise Cohen's d.

    Parameters
    ----------
    essays : list[dict]
        Output of ``test.load_essays``.
    metrics_lookup : dict[int, dict[str, list]]
        Precomputed ``get_all_metrics`` per essay index. Caller is responsible
        for caching; this function never re-runs stylometry.

    Returns
    -------
    dict
        Structured payload with ``metric_weights`` (per-metric float) plus
        diagnostics: counts of same/different pairs, per-metric d_S / d_D / sep
        statistics, and a small debug summary.
    """
    grouped = group_by_author(essays)
    n_essays = len(essays)
    pairs: list[tuple[int, int, str]] = []
    for a in range(n_essays):
        for b in range(a + 1, n_essays):
            label = "same" if grouped.get(author_name(essays[a])) is not None and \
                author_name(essays[a]) == author_name(essays[b]) else "diff"
            pairs.append((a, b, label))

    same_pairs = [p for p in pairs if p[2] == "same"]
    diff_pairs = [p for p in pairs if p[2] == "diff"]

    d_by_metric: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"same": [], "diff": []}
    )

    for i, j, label in pairs:
        m_i = metrics_lookup.get(i)
        m_j = metrics_lookup.get(j)
        if m_i is None or m_j is None:
            continue
        common = sorted(set(m_i) & set(m_j))
        for name in common:
            arr_i = _pair_metric_array(m_i, name)
            arr_j = _pair_metric_array(m_j, name)
            if arr_i is None or arr_j is None:
                continue
            d = cohens_d(arr_i, arr_j)
            if d is None:
                continue
            d_by_metric[name][label].append(float(d))

    stats: dict[str, dict] = {}
    for name, buckets in d_by_metric.items():
        same = buckets["same"]
        diff = buckets["diff"]
        d_S = float(np.mean(same)) if same else 0.0
        d_D = float(np.mean(diff)) if diff else 0.0
        sep = (d_D + EPS) / (d_S + EPS)
        stats[name] = {
            "n_same_pairs": len(same),
            "n_diff_pairs": len(diff),
            "d_S_mean": d_S,
            "d_D_mean": d_D,
            "sep": float(sep),
        }

    if not stats:
        weights: dict[str, float] = {}
    else:
        sep_only_diff = [
            s["sep"] for s in stats.values()
            if s["n_diff_pairs"] >= 1 and s["sep"] > 1.0
        ]
        weights: dict[str, float] = {}
        if sep_only_diff:
            anchor = float(np.mean(sep_only_diff))
            for name, s in stats.items():
                if s["n_diff_pairs"] >= 1 and s["sep"] > 1.0:
                    w = s["sep"] / anchor
                    weights[name] = max(CLAMP_MIN, float(w))
                else:
                    weights[name] = 1.0
        else:
            for name in stats:
                weights[name] = 1.0

    return {
        "schema_version": METRIC_WEIGHTS_SCHEMA_VERSION,
        "method": "cohens_d_pair_ratio",
        "n_same_pairs": len(same_pairs),
        "n_diff_pairs": len(diff_pairs),
        "eps": EPS,
        "clamp_min": CLAMP_MIN,
        "anchor": {
            "metric_weights_mean": float(np.mean(sep_only_diff)) if sep_only_diff else None,
        },
        "metric_weights": {k: float(weights[k]) for k in sorted(weights)},
        "stats": {k: stats[k] for k in sorted(stats)},
    }


def _save_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _print_summary(payload: dict, weights_path: str) -> None:
    weights = payload["metric_weights"]
    if not weights:
        print("[summary] no metric weights produced (no evaluable metrics)")
        return
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    print(f"[summary] wrote {len(weights)} metric weights to {weights_path}")
    print(f"  same-author pairs: {payload['n_same_pairs']}  "
          f"different-author pairs: {payload['n_diff_pairs']}")
    print(f"  anchor (mean sep over >1 metrics): "
          f"{payload['anchor']['metric_weights_mean']}")
    print("  top 5 (highest weight):")
    for name, w in ranked[:5]:
        print(f"    {name:<38} w={w:.3f}")
    print("  bottom 5 (lowest weight):")
    for name, w in ranked[-5:]:
        print(f"    {name:<38} w={w:.3f}")


def _prune_debug(payload: dict, keep_stats_only: bool = True) -> dict:
    """Return a debug-friendly subset (no full weight breakdown needed)."""
    debug = {
        "schema_version": payload["schema_version"],
        "method": payload["method"],
        "n_same_pairs": payload["n_same_pairs"],
        "n_diff_pairs": payload["n_diff_pairs"],
        "stats": payload["stats"],
    }
    if not keep_stats_only:
        debug["metric_weights"] = payload["metric_weights"]
    return debug


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic per-metric weights from pairwise Cohen's d "
            "between essays."
        ),
    )
    parser.add_argument(
        "--essays",
        default=DEFAULT_ESSAYS_PATH,
        help=f"Path to essays.json (default: {DEFAULT_ESSAYS_PATH}).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Where to write metric_weights.json (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip the metric_weights_debug.json sidecar.",
    )
    args = parser.parse_args(argv)

    with open(args.essays, encoding="utf-8") as f:
        essays = json.load(f)
    if len(essays) < 2:
        raise SystemExit("need at least 2 essays to compute pairwise weights")

    print(f"[tune] loading {len(essays)} essays from {args.essays}")
    metrics_lookup: dict[int, dict[str, list]] = {}
    for i, e in enumerate(essays):
        print(f"[tune] computing metrics for essay {i + 1}/{len(essays)}")
        metrics_lookup[i] = get_all_metrics(e["body"])

    payload = compute_metric_weights(essays, metrics_lookup=metrics_lookup)

    os.makedirs(args.out_dir, exist_ok=True)
    weights_path = os.path.join(args.out_dir, DEFAULT_METRIC_WEIGHTS_FILENAME)
    _save_json_atomic(weights_path, payload)
    _print_summary(payload, weights_path)

    if not args.no_debug:
        debug_path = os.path.join(args.out_dir, DEBUG_FILENAME)
        _save_json_atomic(debug_path, _prune_debug(payload, keep_stats_only=False))
        print(f"[tune] debug sidecar written to {debug_path}")


if __name__ == "__main__":
    main()

# Deterministic per-metric weighting derived from KS distances on essay pairs.
#
# Approach:
#   1. Load every essay from essays.json and group by author.
#   2. Enumerate all unordered essay pairs (i, j); tag each as same-author or
#      different-author.
#   3. For each pair, compute KS distance per metric over the same distributions
#      used by the simple global-similarity scorer.
#   4. Aggregate per metric: mean KS across same-author pairs (ks_S) and across
#      different-author pairs (ks_D).
#   5. Separation score: sep = (ks_D + eps) / (ks_S + eps). Higher sep means a
#      metric discriminates authors more than it varies within one author.
#   6. Cube each separation score, normalize around 1.0, and clamp the floor.
#      This sharply promotes author-discriminative metrics while suppressing
#      metrics that vary as much or more within an author.
#   7. Persist to <out_dir>/metric_weights.json plus a debug sidecar.
#
# CLI:
#   python tune.py                                # default plots dir, regex
#   python tune.py --out-dir DIR --essays PATH    # custom output / corpus
#   python tune.py --no-debug                     # skip debug sidecar

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    DEFAULT_METRIC_WEIGHTS_FILENAME,
    METRIC_WEIGHTS_SCHEMA_VERSION,
    _normalize_metric_array,
    _sentence_count,
    get_all_metrics,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
DEBUG_FILENAME = "metric_weights_debug.json"
EPS = 1e-6
CLAMP_MIN = 0.1
WEIGHT_POWER = 3.0


def author_name(essay: dict) -> str:
    return essay["author"].replace("By ", "").strip()


def group_by_author(essays: list[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for i, e in enumerate(essays):
        grouped.setdefault(author_name(e), []).append(i)
    return grouped
def ks_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    return float(ks_2samp(a, b).statistic)



def compute_metric_weights(
    essays: list[dict],
    *,
    metrics_lookup: dict[int, dict[str, list]],
) -> dict:
    """Compute per-metric weights from pairwise KS distances.

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
        diagnostics: counts of same/different pairs, per-metric ks_S / ks_D / sep
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

    distance_by_metric: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"same": [], "diff": []}
    )

    for i, j, label in pairs:
        metrics_i = metrics_lookup.get(i)
        metrics_j = metrics_lookup.get(j)
        if metrics_i is None or metrics_j is None:
            continue
        n_i = _sentence_count(metrics_i) or 0
        n_j = _sentence_count(metrics_j) or 0
        for name in sorted(set(metrics_i) | set(metrics_j)):
            arr_i, arr_j, _ = _normalize_metric_array(
                metrics_i.get(name),
                metrics_j.get(name),
                n_i,
                n_j,
            )
            if arr_i is None or np.std(np.concatenate([arr_i, arr_j])) == 0:
                continue
            distance = ks_distance(arr_i, arr_j)
            if distance is None:
                continue
            distance_by_metric[name][label].append(distance)

    stats: dict[str, dict] = {}
    for name, buckets in distance_by_metric.items():
        same = buckets["same"]
        diff = buckets["diff"]
        ks_S = float(np.mean(same)) if same else 0.0
        ks_D = float(np.mean(diff)) if diff else 0.0
        sep = (ks_D + EPS) / (ks_S + EPS)
        stats[name] = {
            "n_same_pairs": len(same),
            "n_diff_pairs": len(diff),
            "ks_S_mean": ks_S,
            "ks_D_mean": ks_D,
            "sep": float(sep),
        }

    valid_stats = {
        name: stat
        for name, stat in stats.items()
        if stat["n_same_pairs"] >= 1 and stat["n_diff_pairs"] >= 1
    }
    if not valid_stats:
        weights: dict[str, float] = {}
        anchor = None
    else:
        powered_separations = {
            name: stat["sep"] ** WEIGHT_POWER
            for name, stat in valid_stats.items()
        }
        anchor = float(np.mean(list(powered_separations.values())))
        weights = {
            name: max(CLAMP_MIN, powered_separations[name] / anchor)
            if name in powered_separations else CLAMP_MIN
            for name in stats
        }

    return {
        "schema_version": METRIC_WEIGHTS_SCHEMA_VERSION,
        "method": "ks_pair_ratio_power",
        "n_same_pairs": len(same_pairs),
        "n_diff_pairs": len(diff_pairs),
        "eps": EPS,
        "clamp_min": CLAMP_MIN,
        "weight_power": WEIGHT_POWER,
        "anchor": {
            "powered_separation_mean": anchor,
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
    print(f"  weight power: {payload['weight_power']}")
    print(f"  anchor (mean powered separation): "
          f"{payload['anchor']['powered_separation_mean']}")
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
        "weight_power": payload["weight_power"],
        "anchor": payload["anchor"],
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
            "Compute deterministic per-metric weights from pairwise KS "
            "distances between essays."
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

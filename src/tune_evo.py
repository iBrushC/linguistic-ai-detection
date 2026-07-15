#!/usr/bin/env python3
"""
Evolutionary per-metric weighting.

Approach:
  1. Load essays and reuse the cached per-essay metrics when available.
  2. Build the same pair list that test.build_similarity_matrix uses:
       - diagonal: leave-one-out permutations per author
       - off-diagonal: concat-vs-concat per unordered author pair
  3. For each pair, precompute the per-metric (similarity, base_weight) values
     that _global_simple in analysis.py would compute (capped linear weight
     * (1 - KS distance)). This is the only expensive step and runs once.
  4. Evolve a real-valued weight vector with population 25, BLX-alpha
     crossover, Gaussian mutation, tournament selection, elitism=2. Fitness
     is mean(diagonal) - mean(off-diagonal) of the author similarity matrix.
  5. Persist the best weight vector to <out_dir>/metric_weights.json with the
     same schema_version as the Cohen's-d method and a different method tag,
     plus an optional debug sidecar.

CLI:
  python tune_evo.py
  python tune_evo.py --generations 100 --patience 20 --seed 42
  python tune_evo.py --out-dir DIR --essays PATH --no-debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    DEFAULT_METRIC_WEIGHTS_FILENAME,
    METRIC_WEIGHTS_SCHEMA_VERSION,
    get_all_metrics,
    global_similarity,
)
from stylometry import get_sentence_lengths


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
CACHE_FILENAME = ".essay_metrics_cache.json"
DEBUG_FILENAME = "metric_weights_evo_debug.json"

WEIGHT_MIN = 0.1
WEIGHT_MAX = 5.0

DEFAULT_POPULATION = 25
DEFAULT_GENERATIONS = 5000
DEFAULT_PATIENCE = 100
DEFAULT_SEED = 42

TOURNAMENT_SIZE = 3
BLX_ALPHA = 0.5
MUTATION_SIGMA_FRAC = 0.1
MUTATION_PROB = 0.3
ELITE_COUNT = 2

MIN_APPEARANCES = 10


def author_name(essay: dict) -> str:
    return essay["author"].replace("By ", "").strip()


def group_by_author(essays: list[dict]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for i, e in enumerate(essays):
        grouped.setdefault(author_name(e), []).append(i)
    return grouped


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_metric_cache(out_dir: str) -> dict:
    return _load_json(os.path.join(out_dir, CACHE_FILENAME)) or {}


def _save_metric_cache(out_dir: str, cache: dict) -> None:
    _save_json_atomic(os.path.join(out_dir, CACHE_FILENAME), cache)


def _ensure_metrics(
    text: str,
    cache: dict,
    out_dir: str,
) -> dict[str, list]:
    import hashlib

    fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if fp in cache:
        return cache[fp]
    metrics = get_all_metrics(text)
    cache[fp] = metrics
    _save_metric_cache(out_dir, cache)
    return metrics


def _concat_bodies(essays: list[dict], indices: Iterable[int]) -> str:
    return "\n\n".join(essays[i]["body"] for i in indices)


def _collect_metric_names(per_pair: list[dict]) -> list[str]:
    seen: set[str] = set()
    for by_metric in per_pair:
        seen.update(by_metric)
    return sorted(seen)


def _precompute_pair_data(
    essays: list[dict],
    grouped: dict[str, list[int]],
    cache: dict,
    out_dir: str,
) -> tuple[list[dict], list[bool], list[str]]:
    authors = sorted(grouped.keys())
    author_corpus: dict[str, dict[str, list]] = {}
    loo_metrics: dict[str, list[tuple[dict[str, list], dict[str, list]]]] = {}

    print(f"[precompute] {len(authors)} authors: {authors}")
    for author in authors:
        indices = grouped[author]
        corpus_text = _concat_bodies(essays, indices)
        author_corpus[author] = _ensure_metrics(corpus_text, cache, out_dir)

        perms: list[tuple[dict[str, list], dict[str, list]]] = []
        for held_out in indices:
            corpus_idx = [i for i in indices if i != held_out]
            corpus_text_loo = _concat_bodies(essays, corpus_idx)
            held_text = essays[held_out]["body"]
            corpus_m = _ensure_metrics(corpus_text_loo, cache, out_dir)
            held_m = _ensure_metrics(held_text, cache, out_dir)
            perms.append((corpus_m, held_m))
        loo_metrics[author] = perms
        print(f"[precompute] {author}: {len(indices)} essays, {len(perms)} LOO pairs")

    pair_data: list[dict[str, tuple[float, float]]] = []
    is_diag: list[bool] = []

    for author in authors:
        for corpus_m, held_m in loo_metrics[author]:
            pair_data.append(_extract_pair_metrics(corpus_m, held_m))
            is_diag.append(True)

    for i, a in enumerate(authors):
        for j, b in enumerate(authors):
            if j <= i:
                continue
            pair_data.append(_extract_pair_metrics(author_corpus[a], author_corpus[b]))
            is_diag.append(False)

    metric_names = _collect_metric_names(pair_data)
    print(
        f"[precompute] {len(pair_data)} pairs total "
        f"(diag={sum(is_diag)}, off={len(is_diag) - sum(is_diag)}); "
        f"{len(metric_names)} unique metrics"
    )
    return pair_data, is_diag, metric_names


def _extract_pair_metrics(
    metrics_a: dict[str, list],
    metrics_b: dict[str, list],
    min_appearances: int = MIN_APPEARANCES,
) -> dict[str, tuple[float, float]]:
    n_a = len(metrics_a.get("sentence_lengths") or []) or _fallback_n(metrics_a)
    n_b = len(metrics_b.get("sentence_lengths") or []) or _fallback_n(metrics_b)

    out: dict[str, tuple[float, float]] = {}
    for name in sorted(set(metrics_a) | set(metrics_b)):
        a_vals, b_vals, skip = _aligned_arrays(
            metrics_a.get(name), metrics_b.get(name), n_a, n_b
        )
        if a_vals is None or skip:
            continue
        combined = np.concatenate([a_vals, b_vals])
        if float(combined.std()) == 0.0:
            continue
        from scipy.stats import ks_2samp

        ks_stat, _ = ks_2samp(a_vals, b_vals)
        ks_stat = float(ks_stat)
        combined_nz = int(np.count_nonzero(combined))
        base_w = float(min(1.0, combined_nz / min_appearances)) if min_appearances > 0 else 1.0
        similarity = max(0.0, 1.0 - ks_stat)
        out[name] = (similarity, base_w)
    return out


def _fallback_n(metrics: dict[str, list]) -> int:
    counts: dict[int, int] = {}
    for values in metrics.values():
        n = len(values)
        if n:
            counts[n] = counts.get(n, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else 0


def _aligned_arrays(
    values_a, values_b, n_a: int, n_b: int
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
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


def _fitness(
    weights_vec: np.ndarray,
    name_to_idx: dict[str, int],
    pair_data: list[dict],
    is_diag: list[bool],
) -> float:
    diag_vals: list[float] = []
    off_vals: list[float] = []
    for pair_idx, by_metric in enumerate(pair_data):
        num = 0.0
        den = 0.0
        for name, (sim, base_w) in by_metric.items():
            w = base_w * float(weights_vec[name_to_idx[name]])
            if w > 0.0:
                num += w * sim
                den += w
        s = num / den if den > 0.0 else 1.0
        if is_diag[pair_idx]:
            diag_vals.append(s)
        else:
            off_vals.append(s)
    if not diag_vals or not off_vals:
        return float("-inf")
    return float(np.mean(diag_vals) - np.mean(off_vals))


def _seed_population(
    metric_names: list[str],
    seed_weights: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    elite = np.array(
        [float(seed_weights.get(name, 1.0)) for name in metric_names],
        dtype=float,
    )
    elite = np.clip(elite, WEIGHT_MIN, WEIGHT_MAX)

    pop = np.empty((n, len(metric_names)), dtype=float)
    pop[0] = elite

    span = WEIGHT_MAX - WEIGHT_MIN
    sigma = 0.15 * span
    for i in range(1, n):
        noise = rng.normal(loc=0.0, scale=sigma, size=len(metric_names))
        pop[i] = np.clip(elite + noise, WEIGHT_MIN, WEIGHT_MAX)
    return pop


def _tournament_select(
    fitnesses: np.ndarray,
    rng: np.random.Generator,
) -> int:
    contenders = rng.integers(0, len(fitnesses), size=TOURNAMENT_SIZE)
    best = contenders[0]
    for c in contenders[1:]:
        if fitnesses[c] > fitnesses[best]:
            best = c
    return int(best)


def _blx_alpha_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    span = hi - lo
    low = lo - BLX_ALPHA * span
    high = hi + BLX_ALPHA * span
    c1 = rng.uniform(low, high)
    c2 = rng.uniform(low, high)
    c1 = np.clip(c1, WEIGHT_MIN, WEIGHT_MAX)
    c2 = np.clip(c2, WEIGHT_MIN, WEIGHT_MAX)
    return c1, c2


def _mutate(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    span = WEIGHT_MAX - WEIGHT_MIN
    sigma = MUTATION_SIGMA_FRAC * span
    mask = rng.random(size=ind.shape) < MUTATION_PROB
    noise = rng.normal(loc=0.0, scale=sigma, size=ind.shape)
    return np.clip(ind + mask * noise, WEIGHT_MIN, WEIGHT_MAX)


def _evaluate_population(
    pop: np.ndarray,
    name_to_idx: dict[str, int],
    pair_data: list[dict],
    is_diag: list[bool],
) -> np.ndarray:
    return np.array(
        [_fitness(ind, name_to_idx, pair_data, is_diag) for ind in pop],
        dtype=float,
    )


def _run_ga(
    pop: np.ndarray,
    name_to_idx: dict[str, int],
    pair_data: list[dict],
    is_diag: list[bool],
    generations: int,
    patience: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], int]:
    fitnesses = _evaluate_population(pop, name_to_idx, pair_data, is_diag)
    best_idx = int(np.argmax(fitnesses))
    best_ind = pop[best_idx].copy()
    best_fit = float(fitnesses[best_idx])

    best_history: list[float] = [best_fit]
    mean_history: list[float] = [float(fitnesses.mean())]

    stagnant = 0
    generations_run = 0
    for gen in range(1, generations + 1):
        generations_run = gen
        offspring: list[np.ndarray] = []
        while len(offspring) < len(pop):
            i1 = _tournament_select(fitnesses, rng)
            i2 = _tournament_select(fitnesses, rng)
            c1, c2 = _blx_alpha_crossover(pop[i1], pop[i2], rng)
            offspring.append(_mutate(c1, rng))
            if len(offspring) < len(pop):
                offspring.append(_mutate(c2, rng))
        offspring_arr = np.array(offspring[: len(pop)], dtype=float)

        off_fitnesses = _evaluate_population(
            offspring_arr, name_to_idx, pair_data, is_diag
        )

        combined = np.vstack([pop, offspring_arr])
        combined_fit = np.concatenate([fitnesses, off_fitnesses])
        order = np.argsort(combined_fit)[::-1]
        pop = combined[order[: len(pop)]]
        fitnesses = combined_fit[order[: len(pop)]]

        gen_best = float(fitnesses[0])
        gen_mean = float(fitnesses.mean())
        best_history.append(max(best_history[-1], gen_best))
        mean_history.append(gen_mean)

        if gen_best > best_fit + 1e-12:
            best_fit = gen_best
            best_ind = pop[0].copy()
            stagnant = 0
        else:
            stagnant += 1

        print(
            f"[gen {gen:>3}] best={gen_best:.4f}  mean={gen_mean:.4f}  "
            f"overall_best={best_fit:.4f}  stagnant={stagnant}"
        )

        if stagnant >= patience:
            print(f"[gen {gen}] early stop: no improvement for {patience} generations")
            break

    return best_ind, pop, best_history, mean_history, generations_run


def _load_seed_weights(out_dir: str, metric_names: list[str]) -> dict[str, float]:
    payload = _load_json(os.path.join(out_dir, DEFAULT_METRIC_WEIGHTS_FILENAME))
    seed: dict[str, float] = {name: 1.0 for name in metric_names}
    if isinstance(payload, dict):
        existing = payload.get("metric_weights")
        if isinstance(existing, dict):
            for name, value in existing.items():
                try:
                    seed[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
    return seed


def _diag_off_for_weights(
    weights_vec: np.ndarray,
    name_to_idx: dict[str, int],
    pair_data: list[dict],
    is_diag: list[bool],
) -> tuple[float, float, int, int]:
    diag_vals: list[float] = []
    off_vals: list[float] = []
    for pair_idx, by_metric in enumerate(pair_data):
        num = 0.0
        den = 0.0
        for name, (sim, base_w) in by_metric.items():
            w = base_w * float(weights_vec[name_to_idx[name]])
            if w > 0.0:
                num += w * sim
                den += w
        s = num / den if den > 0.0 else 1.0
        if is_diag[pair_idx]:
            diag_vals.append(s)
        else:
            off_vals.append(s)
    return (
        float(np.mean(diag_vals)) if diag_vals else float("nan"),
        float(np.mean(off_vals)) if off_vals else float("nan"),
        len(diag_vals),
        len(off_vals),
    )


def _print_summary(
    weights: dict[str, float],
    diag: float,
    off: float,
    n_diag: int,
    n_off: int,
    generations: int,
    weights_path: str,
) -> None:
    gap = diag - off
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    print(
        f"[summary] wrote {len(weights)} metric weights to {weights_path}\n"
        f"  generations run: {generations}\n"
        f"  diagonal mean: {diag:.4f}  (n={n_diag})\n"
        f"  off-diagonal mean: {off:.4f}  (n={n_off})\n"
        f"  fitness gap: {gap:+.4f}\n"
        f"  top 5 (highest weight):"
    )
    for name, w in ranked[:5]:
        print(f"    {name:<38} w={w:.3f}")
    print("  bottom 5 (lowest weight):")
    for name, w in ranked[-5:]:
        print(f"    {name:<38} w={w:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evolve per-metric weights with a simple real-valued GA. Fitness "
            "is mean(diagonal) - mean(off-diagonal) of the author similarity "
            "matrix; the diagonal is LOO self-similarity and off-diagonal is "
            "concat-vs-concat cross similarity."
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
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
        help=f"GA population size (default: {DEFAULT_POPULATION}).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help=f"Max GA generations (default: {DEFAULT_GENERATIONS}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=(
            "Stop after this many consecutive generations with no improvement "
            f"(default: {DEFAULT_PATIENCE})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip the metric_weights_evo_debug.json sidecar.",
    )
    args = parser.parse_args(argv)

    with open(args.essays, encoding="utf-8") as f:
        essays = json.load(f)
    if len(essays) < 4:
        raise SystemExit("need at least 4 essays (>=2 authors, >=2 essays each)")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    grouped = group_by_author(essays)
    counts = {a: len(idx) for a, idx in grouped.items()}
    bad = [a for a, idx in grouped.items() if len(idx) < 2]
    if bad:
        raise SystemExit(
            f"authors with <2 essays cannot run LOO: {bad}; "
            "need at least 2 essays per author."
        )
    if len(grouped) < 2:
        raise SystemExit("need at least 2 distinct authors for cross similarity")
    print(
        f"[init] loaded {len(essays)} essays across {len(grouped)} authors: {counts}"
    )

    cache = _load_metric_cache(out_dir)
    pair_data, is_diag, metric_names = _precompute_pair_data(
        essays, grouped, cache, out_dir
    )
    name_to_idx = {name: i for i, name in enumerate(metric_names)}

    seed_weights = _load_seed_weights(out_dir, metric_names)
    rng = np.random.default_rng(args.seed)
    pop = _seed_population(metric_names, seed_weights, args.population, rng)

    initial_best_idx = int(np.argmax(_evaluate_population(
        pop, name_to_idx, pair_data, is_diag
    )))
    initial_best = pop[initial_best_idx]
    initial_diag, initial_off, initial_n_diag, initial_n_off = _diag_off_for_weights(
        initial_best, name_to_idx, pair_data, is_diag
    )
    initial_fitness = (
        initial_diag - initial_off
        if not (np.isnan(initial_diag) or np.isnan(initial_off))
        else float("-inf")
    )
    print(
        f"[init] seed fitness (elite individual) = {initial_fitness:.4f} "
        f"(diag={initial_diag:.4f}, off={initial_off:.4f})"
    )

    best_ind, final_pop, best_hist, mean_hist, gens_run = _run_ga(
        pop,
        name_to_idx,
        pair_data,
        is_diag,
        args.generations,
        args.patience,
        rng,
    )

    final_fitnesses = _evaluate_population(
        final_pop, name_to_idx, pair_data, is_diag
    )
    final_diag, final_off, final_n_diag, final_n_off = _diag_off_for_weights(
        best_ind, name_to_idx, pair_data, is_diag
    )
    final_fitness = final_diag - final_off

    weights = {
        name: float(best_ind[name_to_idx[name]]) for name in metric_names
    }
    weights_sorted = {k: weights[k] for k in sorted(weights)}

    payload = {
        "schema_version": METRIC_WEIGHTS_SCHEMA_VERSION,
        "method": "evolutionary_ga",
        "metric_weights": weights_sorted,
        "metric_bounds": {"min": WEIGHT_MIN, "max": WEIGHT_MAX},
        "population_size": args.population,
        "generations_run": gens_run,
        "patience_used": args.patience,
        "ga_hyperparameters": {
            "tournament_size": TOURNAMENT_SIZE,
            "blx_alpha": BLX_ALPHA,
            "mutation_sigma_frac": MUTATION_SIGMA_FRAC,
            "mutation_prob": MUTATION_PROB,
            "elite_count": ELITE_COUNT,
            "min_appearances": MIN_APPEARANCES,
        },
        "fitness": {
            "initial": float(initial_fitness),
            "final": float(final_fitness),
            "improvement": float(final_fitness - initial_fitness),
            "diagonal_mean": final_diag,
            "off_diagonal_mean": final_off,
            "n_diag_pairs": final_n_diag,
            "n_off_pairs": final_n_off,
        },
        "history": {
            "best_per_generation": [float(x) for x in best_hist],
            "mean_per_generation": [float(x) for x in mean_hist],
        },
        "seed": args.seed,
    }

    weights_path = os.path.join(out_dir, DEFAULT_METRIC_WEIGHTS_FILENAME)
    _save_json_atomic(weights_path, payload)
    _print_summary(
        weights_sorted,
        final_diag,
        final_off,
        final_n_diag,
        final_n_off,
        gens_run,
        weights_path,
    )

    if not args.no_debug:
        debug_payload = {
            "schema_version": METRIC_WEIGHTS_SCHEMA_VERSION,
            "method": "evolutionary_ga",
            "ga_hyperparameters": payload["ga_hyperparameters"],
            "metric_bounds": payload["metric_bounds"],
            "fitness": payload["fitness"],
            "history": payload["history"],
            "seed_weights_loaded": seed_weights,
            "initial_best_weights": {
                name: float(initial_best[name_to_idx[name]]) for name in metric_names
            },
            "final_best_weights": weights_sorted,
            "final_population_fitness": [float(x) for x in final_fitnesses],
            "rng_seed": args.seed,
        }
        debug_path = os.path.join(out_dir, DEBUG_FILENAME)
        _save_json_atomic(debug_path, debug_payload)
        print(f"[tune_evo] debug sidecar written to {debug_path}")


if __name__ == "__main__":
    main()
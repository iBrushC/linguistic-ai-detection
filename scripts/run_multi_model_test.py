"""Multi-model OpenRouter cross-validation driver.

Runs assignment, recreation, and leakage-safe Cosine Delta scoring for every
alias in ``models.json``, reusing existing artifacts and resuming partial work.

Usage::

    python scripts/run_multi_model_test.py
    python scripts/run_multi_model_test.py --dry-run
    python scripts/run_multi_model_test.py --models glm-5.2 chatgpt-5.6
    python scripts/run_multi_model_test.py --model-workers 4 --models-concurrency 2
    python scripts/run_multi_model_test.py --force

What it does:

1. Reuses ``generated/experiment_manifest.json`` (regenerates only if missing).
2. Back-fills ``generated/essay_details.json`` if any selected essays are
   unassigned, using the first requested model.
3. For each alias in ``models.json`` (or the explicit ``--models`` list):
   - Skips recreation if its ``essays_<alias>_recreate.json`` already covers
     every selected essay, unless ``--force`` is set.
   - Otherwise invokes ``src.generate recreate`` with the per-model worker pool.
4. Scores each completed model under ``src/plots/experiment_<alias>/``.
5. Writes a cross-model summary to ``src/plots/experiment_multi/summary.json``.

Set ``OPENROUTER_API_KEY`` before running without ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import experiment as experiment_mod  # noqa: E402
from src import generate as generate_mod  # noqa: E402


GENERATED_DIR = os.path.join(REPO_ROOT, "generated")
DETAILS_FILENAME = "essay_details.json"


def details_path(generated_dir: str) -> str:
    return os.path.join(generated_dir, DETAILS_FILENAME)

DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "src", "plots")
DEFAULT_SEED = 1729
N_AUTHORS = 5
N_ESSAYS_PER_AUTHOR = 5


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: str, obj) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=parent or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def recreation_path(alias: str, generated_dir: str) -> str:
    return os.path.join(generated_dir, f"essays_{alias}_recreate.json")


def plot_dir_for(alias: str, root: str) -> str:
    return os.path.join(root, f"experiment_{alias}")


def multi_plot_dir(root: str) -> str:
    return os.path.join(root, "experiment_multi")


def manifest_essay_ids(manifest: dict) -> set[str]:
    return {e["article_id"] for e in manifest.get("essays", [])}


def is_complete(path: str, target_ids: set[str], *, key: str = "source_article_id") -> bool:
    if not os.path.exists(path):
        return False
    data = load_json(path, default=[])
    if not isinstance(data, list):
        return False
    have: set[str] = set()
    for item in data:
        if not item:
            continue
        v = item.get(key) or item.get("article_id")
        if v:
            have.add(str(v))
    have.discard(None)  # type: ignore[arg-type]
    return target_ids.issubset(have)


def ensure_manifest(args, logger=print) -> dict:
    path = os.path.join(args.generated_dir, experiment_mod.MANIFEST_FILENAME)
    if args.no_select and os.path.exists(path):
        manifest = load_json(path, default={})
        logger(f"[manifest] reused {path} ({len(manifest.get('essays', []))} essays)")
        return manifest
    manifest = experiment_mod.build_manifest(
        seed=args.seed,
        n_authors=N_AUTHORS,
        n_essays_per_author=N_ESSAYS_PER_AUTHOR,
        chunk_size=args.chunk_size,
        min_fill_ratio=args.min_fill_ratio,
        min_chunks_per_essay=args.min_chunks_per_essay,
    )
    save_json_atomic(path, manifest)
    logger(f"[manifest] wrote {path} ({len(manifest['essays'])} essays)")
    return manifest


def ensure_details(args, manifest: dict, target_alias: str, logger=print) -> bool:
    target_ids = manifest_essay_ids(manifest)
    path = details_path(args.generated_dir)
    if is_complete(path, target_ids, key="article_id"):
        logger(f"[details] reused {path}")
        return True
    if args.dry_run:
        _write_stub_details(manifest, path)
        logger(f"[details] wrote dry-run stubs to {path}")
        return True
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger("[details] OPENROUTER_API_KEY not set", file=sys.stderr)
        return False
    rc = generate_mod.main([
        "details",
        "--model", target_alias,
        "--use-selector", os.path.join(args.generated_dir, experiment_mod.MANIFEST_FILENAME),
    ])
    return rc == 0 and is_complete(path, target_ids, key="article_id")


def _write_stub_details(manifest: dict, path: str) -> None:
    essays = load_json(os.path.join(REPO_ROOT, "essays.json"), default=[])
    by_id = {e["article_id"]: e for e in essays}
    ordered: list[dict | None] = []
    for entry in manifest["essays"]:
        article_id = entry["article_id"]
        essay = by_id.get(article_id, {})
        wc = len((essay.get("text") or "").split())
        ordered.append({
            "article_id": article_id,
            "author": entry["author"],
            "title": article_id,
            "word_count": wc,
            "assignment": f"DRY-RUN: write a {wc}-word piece on the topic of {article_id}.",
            "model_slug": "dry-run",
            "model_alias": "dry-run",
        })
    save_json_atomic(path, ordered)


def _write_stub_recreations(manifest: dict, path: str, alias: str) -> None:
    essays = load_json(os.path.join(REPO_ROOT, "essays.json"), default=[])
    by_id = {e["article_id"]: e for e in essays}
    recs: list[dict] = []
    for entry in manifest["essays"]:
        article_id = entry["article_id"]
        source = by_id.get(article_id, {})
        text = source.get("text", "")
        recs.append({
            "article_id": f"{article_id}::gen",
            "source_article_id": article_id,
            "author": entry["author"],
            "title": article_id,
            "body": text or "DRY-RUN fallback body.",
            "model_slug": "dry-run",
            "model_alias": alias,
            "word_count": len(text.split()) if text else 0,
        })
    save_json_atomic(path, recs)


def remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def process_model(alias: str, args, manifest: dict) -> dict:
    log: list[str] = []
    def L(msg: str) -> None:
        line = f"[{alias}] {msg}"
        print(line)
        log.append(line)

    target_ids = manifest_essay_ids(manifest)
    rec_path = recreation_path(alias, args.generated_dir)
    manifest_path = os.path.join(args.generated_dir, experiment_mod.MANIFEST_FILENAME)

    already_complete = is_complete(rec_path, target_ids)
    recreated = False
    if not already_complete or args.force:
        if args.force and os.path.exists(rec_path):
            remove_if_exists(rec_path)
            L(f"--force requested; cleared {rec_path}")
        L(f"recreating via {alias} -> {rec_path}")
        if not ensure_details(args, manifest, alias, L):
            L("details stage failed; aborting model", file=sys.stderr)
            return {"alias": alias, "ok": False, "recreated": False, "scored": False, "log": log}
        if not args.dry_run:
            rc = generate_mod.main([
                "recreate",
                "--model", alias,
                "--use-selector", manifest_path,
                "--workers", str(max(1, args.model_workers)),
                *(["--overwrite"] if args.force else []),
            ])
            if rc != 0:
                L(f"recreate exited rc={rc}", file=sys.stderr)
                return {"alias": alias, "ok": False, "recreated": False, "scored": False, "log": log}
        else:
            _write_stub_recreations(manifest, rec_path, alias)
            L(f"wrote dry-run stubs to {rec_path}")
        recreated = True
    else:
        L(f"recreations already complete at {rec_path}; skipping")

    scored = False
    if not args.no_score:
        if not is_complete(rec_path, target_ids):
            L("recreations incomplete after generation; refusing to score",
              file=sys.stderr)
            return {"alias": alias, "ok": False, "recreated": recreated, "scored": False, "log": log}
        out = plot_dir_for(alias, args.out_root)
        os.makedirs(out, exist_ok=True)
        L(f"scoring into {out}")
        rc = experiment_mod.main([
            "score",
            "--manifest", manifest_path,
            "--recreations", rec_path,
            "--model", alias,
            "--out-dir", out,
            "--chunk-size", str(args.chunk_size),
            "--min-fill-ratio", str(args.min_fill_ratio),
            "--mfw-n", str(args.mfw_n),
        ])
        if rc != 0:
            L(f"score exited rc={rc}", file=sys.stderr)
            return {"alias": alias, "ok": False, "recreated": recreated, "scored": False, "log": log}
        scored = True

    return {"alias": alias, "ok": True, "recreated": recreated, "scored": scored, "log": log}


def write_multi_summary(args, aliases: Sequence[str]) -> dict | None:
    root = multi_plot_dir(args.out_root)
    os.makedirs(root, exist_ok=True)
    per_model: dict[str, dict] = {}
    for alias in aliases:
        summary_path = os.path.join(plot_dir_for(alias, args.out_root), "summary.json")
        if os.path.exists(summary_path):
            per_model[alias] = load_json(summary_path, default={})
    if not per_model:
        return None
    cross = {
        "overall_delta_mean": {
            alias: data.get("overall", {}).get("delta_mean")
            for alias, data in per_model.items()
        },
        "per_author_delta_mean": {
            alias: {author: stats.get("delta_mean") for author, stats in data.get("per_author", {}).items()}
            for alias, data in per_model.items()
        },
    }
    out = {"aliases": sorted(per_model), "per_model": per_model, "cross": cross}
    save_json_atomic(os.path.join(root, "summary.json"), out)
    plot_path = _plot_comparison(per_model, root)
    if plot_path:
        out["comparison_plot"] = plot_path
        save_json_atomic(os.path.join(root, "summary.json"), out)
    return out


def _plot_comparison(per_model: dict[str, dict], out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    aliases = sorted(per_model)
    authors = sorted({a for data in per_model.values() for a in data.get("per_author", {})})
    if not aliases or not authors:
        return None

    deltas = np.full((len(aliases), len(authors)), np.nan)
    for i, alias in enumerate(aliases):
        per_author = per_model[alias].get("per_author", {})
        for j, author in enumerate(authors):
            stats = per_author.get(author, {})
            v = stats.get("delta_mean")
            if v is not None:
                deltas[i, j] = float(v)

    width = 0.8 / max(1, len(aliases))
    x = np.arange(len(authors))
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, alias in enumerate(aliases):
        offset = (i - (len(aliases) - 1) / 2) * width
        ax.bar(x + offset, deltas[i], width, label=alias, color=cmap(i % 10), edgecolor="black")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(authors, rotation=25, ha="right")
    ax.set_ylabel("mean Cosine Delta (AI - natural)")
    ax.set_title("Cosine Delta delta across models")
    ax.legend(loc="best")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "comparison_cosine_delta.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-model OpenRouter cross-validation driver.",
    )
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of models.json aliases; defaults to all.")
    p.add_argument("--skip-models", nargs="*", default=[],
                   help="Aliases to exclude entirely.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--chunk-size", type=int, default=experiment_mod.DEFAULT_CHUNK_SIZE)
    p.add_argument("--min-fill-ratio", type=float, default=experiment_mod.DEFAULT_MIN_FILL_RATIO)
    p.add_argument("--min-chunks-per-essay", type=int, default=1)
    p.add_argument("--mfw-n", type=int, default=experiment_mod.DEFAULT_MFW_N)
    p.add_argument("--generated-dir", default=GENERATED_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip OpenRouter calls; stubs are still scored offline.")
    p.add_argument("--no-select", action="store_true",
                   help="Reuse an existing experiment_manifest.json without rebuilding it.")
    p.add_argument("--no-score", action="store_true",
                   help="Skip the per-model Cosine Delta scoring stage.")
    p.add_argument("--no-summarize", action="store_true",
                   help="Skip the cross-model summary aggregation.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate recreations and re-score even when artifacts exist.")
    p.add_argument("--model-workers", type=int, default=1,
                   help="Per-model OpenRouter worker pool size (default 1).")
    p.add_argument("--models-concurrency", type=int, default=1,
                   help="Number of models to run in parallel (default 1).")
    return p


def resolve_aliases(args, models_all: dict[str, str]) -> list[str]:
    if args.models:
        aliases = list(args.models)
        missing = [a for a in aliases if a not in models_all]
        if missing:
            print(f"[warn] unknown aliases ignored: {missing}", file=sys.stderr)
        aliases = [a for a in aliases if a in models_all]
    else:
        aliases = sorted(models_all)
    skip = set(args.skip_models or [])
    return [a for a in aliases if a not in skip]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print("[abort] OPENROUTER_API_KEY not set; use --dry-run for offline sanity check",
              file=sys.stderr)
        return 2

    manifest = ensure_manifest(args)
    target_ids = manifest_essay_ids(manifest)
    if not target_ids:
        print("[abort] manifest has no selected essays", file=sys.stderr)
        return 1

    models_all = generate_mod.load_models()
    aliases = resolve_aliases(args, models_all)
    if not aliases:
        print("[abort] no models selected", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"[plan] {len(aliases)} models: {aliases}")
    print(f"[plan] model_workers={args.model_workers}  models_concurrency={args.models_concurrency}")
    print(f"[plan] dry_run={args.dry_run}  force={args.force}  no_score={args.no_score}")
    print("=" * 78)

    results: list[dict] = []
    if args.models_concurrency <= 1:
        for alias in aliases:
            results.append(process_model(alias, args, manifest))
    else:
        with ThreadPoolExecutor(max_workers=args.models_concurrency) as ex:
            futures = {ex.submit(process_model, alias, args, manifest): alias for alias in aliases}
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda r: r["alias"])
    ok_count = sum(1 for r in results if r.get("ok"))

    print()
    print("=" * 78)
    print(f"[done] {ok_count}/{len(results)} models ok; elapsed {time.time() - t0:.1f}s")
    for r in results:
        status = "ok" if r.get("ok") else "FAIL"
        print(f"  - {r['alias']:<18} recreated={r.get('recreated')} scored={r.get('scored')}  {status}")
    print("=" * 78)

    if not args.no_summarize and not args.no_score:
        multi = write_multi_summary(args, [r["alias"] for r in results])
        if multi:
            print(f"[done] wrote cross-model summary to {multi_plot_dir(args.out_root)}/")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

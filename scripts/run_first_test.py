"""Single-command driver for the first OpenRouter cross-validation test.

Usage::

    python scripts/run_first_test.py
    python scripts/run_first_test.py --dry-run           # offline sanity check
    python scripts/run_first_test.py --model glm-5.2     # choose model alias
    python scripts/run_first_test.py --skip-details      # reuse existing assignments
    python scripts/run_first_test.py --skip-recreate     # reuse existing recreations

What it does:

1. Selects five authors and five essays per author (seed 1729) and writes
   ``generated/experiment_manifest.json``.
2. Calls OpenRouter with ``glm-5.2`` (default) to build one shared
   writing assignment per selected essay. Skipped if ``--dry-run`` or
   ``--skip-details`` is set, or if the assignment file already covers
   every selected essay.
3. Calls OpenRouter with the same model to recreate every selected essay
   using its assignment plus the other four natural essays. Skipped if
   ``--dry-run`` or ``--skip-recreate`` is set, or if a recreation file
   already covers every selected essay.
4. Runs the leakage-safe Cosine Delta scoring across the 25 folds and
   produces ``src/plots/experiment_first/{folds.json, summary.json,
   folds_cosine_delta.png}``.

Set ``OPENROUTER_API_KEY`` before running without ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import chunking, experiment, generate as generate_mod  # noqa: E402


GENERATED_DIR = os.path.join(REPO_ROOT, "generated")
DEFAULT_DETAILS_FILENAME = "essay_details.json"
DEFAULT_MODEL_ALIAS = "glm-5.2"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "src", "plots", "experiment_first")
DEFAULT_SEED = 1729
N_AUTHORS = 5
N_ESSAYS_PER_AUTHOR = 5


def _details_path(generated_dir: str) -> str:
    return os.path.join(generated_dir, DEFAULT_DETAILS_FILENAME)


def _recreations_path(model_alias: str, generated_dir: str | None = None) -> str:
    base = generated_dir if generated_dir is not None else GENERATED_DIR
    return os.path.join(base, f"essays_{model_alias}_recreate.json")


# Glob-style discovery for any existing recreate file under generated_dir.
def _find_recreations(generated_dir: str) -> str | None:
    if not os.path.isdir(generated_dir):
        return None
    candidates = sorted(
        os.path.join(generated_dir, n)
        for n in os.listdir(generated_dir)
        if n.startswith("essays_") and n.endswith("_recreate.json")
    )
    return candidates[0] if candidates else None


# --- IO helpers ---------------------------------------------------------


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


# --- Stage 1: select ---------------------------------------------------


def stage_select(args) -> dict:
    print("=" * 78)
    print(f"[1/4] selecting {N_AUTHORS} authors x {N_ESSAYS_PER_AUTHOR} essays "
          f"(seed={args.seed})")
    print("=" * 78)
    os.makedirs(args.generated_dir, exist_ok=True)
    manifest_path = os.path.join(args.generated_dir, experiment.MANIFEST_FILENAME)
    if args.no_select and os.path.exists(manifest_path):
        manifest = load_json(manifest_path, default={})
        print(f"[select] reusing existing manifest at {manifest_path} "
              f"({len(manifest.get('essays', []))} essays)")
        return manifest
    manifest = experiment.build_manifest(
        seed=args.seed,
        n_authors=N_AUTHORS,
        n_essays_per_author=N_ESSAYS_PER_AUTHOR,
        chunk_size=args.chunk_size,
        min_fill_ratio=args.min_fill_ratio,
        min_chunks_per_essay=args.min_chunks_per_essay,
    )
    save_json_atomic(manifest_path, manifest)
    print(f"[select] wrote {manifest_path}")
    print(f"[select] {len(manifest['essays'])} essays across "
          f"{len({e['author'] for e in manifest['essays']})} authors")
    return manifest


# --- Stage 2: assignments ----------------------------------------------


def _has_all_assignments(manifest: dict, path: str) -> bool:
    data = load_json(path, default=[])
    if not isinstance(data, list):
        return False
    have = {item.get("article_id") for item in data if item}
    needed = {e["article_id"] for e in manifest["essays"]}
    return needed.issubset(have)


def stage_details(args, manifest: dict) -> bool:
    print()
    print("=" * 78)
    print(f"[2/4] generating shared assignments via {args.model}")
    print("=" * 78)
    details_path = _details_path(args.generated_dir)
    if args.skip_details:
        print("[details] --skip-details: reusing existing assignments without verification")
        if not load_json(details_path, default=None):
            print(f"[details] no existing details file at {details_path}",
                  file=sys.stderr)
            return False
        return True
    if _has_all_assignments(manifest, details_path):
        print(f"[details] {details_path} already covers all selected essays")
        return True
    if args.dry_run:
        print(f"[details] DRY RUN: writing stub assignments for "
              f"{len(manifest['essays'])} essays")
        _write_stub_assignments(manifest, details_path)
        return True
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[details] OPENROUTER_API_KEY not set; skipping generation",
              file=sys.stderr)
        return False

    models = generate_mod.load_models()
    if args.model not in models:
        print(f"[details] unknown model alias {args.model!r}; choices: "
              f"{sorted(models)}", file=sys.stderr)
        return False
    manifest_path = os.path.join(args.generated_dir, experiment.MANIFEST_FILENAME)
    rc = generate_mod.main([
        "details",
        "--model", args.model,
        "--use-selector", manifest_path,
    ])
    return rc == 0 and _has_all_assignments(manifest, details_path)


def _write_stub_assignments(manifest: dict, details_path: str) -> None:
    essays = load_json(os.path.join(REPO_ROOT, "essays.json"), default=[])
    by_id = {e["article_id"]: e for e in essays}
    stub: list[dict | None] = []
    for entry in manifest["essays"]:
        essay = by_id.get(entry["article_id"], {})
        text = essay.get("text", "")
        wc = len(text.split())
        stub.append({
            "article_id": entry["article_id"],
            "author": entry["author"],
            "title": entry["article_id"],
            "word_count": wc,
            "assignment": f"DRY-RUN: write a {wc}-word piece on the topic of "
                          f"{entry['article_id']}.",
            "model_slug": "dry-run",
            "model_alias": "dry-run",
        })
    save_json_atomic(details_path, stub)
    print(f"[details] wrote {details_path} (dry-run stubs)")


# --- Stage 3: recreations ----------------------------------------------


def _has_all_recreations(manifest: dict, path: str) -> bool:
    data = load_json(path, default=[])
    if not isinstance(data, list):
        return False
    have = {item.get("source_article_id") for item in data if item}
    needed = {e["article_id"] for e in manifest["essays"]}
    return needed.issubset(have)


def stage_recreate(args, manifest: dict) -> bool:
    print()
    print("=" * 78)
    print(f"[3/4] recreating essays via {args.model} with author context")
    print("=" * 78)
    details_path = _details_path(args.generated_dir)
    rec_path = _recreations_path("dry-run" if args.dry_run else args.model,
                                 args.generated_dir)
    if args.skip_recreate:
        existing = _find_recreations(args.generated_dir)
        if existing:
            print(f"[recreate] --skip-recreate: reusing existing recreations at {existing}")
            return True
        print("[recreate] --skip-recreate but no recreations file found",
              file=sys.stderr)
        return False
    if _has_all_recreations(manifest, rec_path):
        print(f"[recreate] {rec_path} already covers all selected essays")
        return True
    if args.dry_run:
        print(f"[recreate] DRY RUN: writing stub recreations for "
              f"{len(manifest['essays'])} essays")
        _write_stub_recreations(manifest, rec_path)
        return True
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[recreate] OPENROUTER_API_KEY not set; skipping generation",
              file=sys.stderr)
        return False
    if not _has_all_assignments(manifest, details_path):
        print("[recreate] missing assignments; run stage 2 first",
              file=sys.stderr)
        return False

    manifest_path = os.path.join(args.generated_dir, experiment.MANIFEST_FILENAME)
    rc = generate_mod.main([
        "recreate",
        "--model", args.model,
        "--use-selector", manifest_path,
    ])
    return rc == 0 and _has_all_recreations(manifest, rec_path)


def _write_stub_recreations(manifest: dict, rec_path: str) -> None:
    essays = load_json(os.path.join(REPO_ROOT, "essays.json"), default=[])
    by_id = {e["article_id"]: e for e in essays}
    recs: list[dict] = []
    for entry in manifest["essays"]:
        source = by_id.get(entry["article_id"], {})
        text = source.get("text", "")
        recs.append({
            "article_id": f"{entry['article_id']}::gen",
            "source_article_id": entry["article_id"],
            "author": entry["author"],
            "title": entry["article_id"],
            "body": text or "DRY-RUN fallback body.",
            "model_slug": "dry-run",
            "model_alias": "dry-run",
            "word_count": len(text.split()) if text else 0,
        })
    save_json_atomic(rec_path, recs)
    print(f"[recreate] wrote {rec_path} (dry-run stubs)")


# --- Stage 4: score ----------------------------------------------------


def stage_score(args, manifest: dict) -> int:
    print()
    print("=" * 78)
    print("[4/4] scoring 25 folds with leakage-safe Cosine Delta")
    print("=" * 78)
    preferred = _recreations_path("dry-run" if args.dry_run else args.model,
                                  args.generated_dir)
    rec_path: str | None = None
    if args.skip_recreate:
        rec_path = _find_recreations(args.generated_dir) or preferred
    elif _has_all_recreations(manifest, preferred):
        rec_path = preferred
    else:
        existing = _find_recreations(args.generated_dir)
        if existing and _has_all_recreations(manifest, existing):
            rec_path = existing
            print(f"[score] using existing recreations at {rec_path}")
    if rec_path is None:
        print(f"[score] recreations missing at {preferred}; cannot score",
              file=sys.stderr)
        return 2

    rc = experiment.main([
        "score",
        "--manifest", os.path.join(args.generated_dir, experiment.MANIFEST_FILENAME),
        "--recreations", rec_path,
        "--model", "dry-run" if args.dry_run else args.model,
        "--out-dir", args.out_dir,
        "--chunk-size", str(args.chunk_size),
        "--min-fill-ratio", str(args.min_fill_ratio),
        "--mfw-n", str(args.mfw_n),
    ])
    return rc


# --- CLI ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the first OpenRouter cross-validation test end-to-end.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ALIAS,
                        help=f"Friendly model name from models.json (default: {DEFAULT_MODEL_ALIAS}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Selection seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--chunk-size", type=int, default=experiment.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-fill-ratio", type=float, default=experiment.DEFAULT_MIN_FILL_RATIO)
    parser.add_argument("--min-chunks-per-essay", type=int, default=1)
    parser.add_argument("--mfw-n", type=int, default=experiment.DEFAULT_MFW_N)
    parser.add_argument("--generated-dir", default=GENERATED_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip OpenRouter calls; selection and scoring logic still run.")
    parser.add_argument("--skip-details", action="store_true",
                        help="Do not call OpenRouter for assignment generation.")
    parser.add_argument("--skip-recreate", action="store_true",
                        help="Do not call OpenRouter for essay recreation.")
    parser.add_argument("--no-select", action="store_true",
                        help="Reuse an existing experiment_manifest.json instead of regenerating it.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    manifest = stage_select(args)
    if not stage_details(args, manifest):
        print("[abort] assignment stage failed", file=sys.stderr)
        return 1
    if not stage_recreate(args, manifest):
        print("[abort] recreation stage failed", file=sys.stderr)
        return 1
    rc = stage_score(args, manifest)

    elapsed = time.time() - t0
    print()
    print("=" * 78)
    print(f"[done] elapsed {elapsed:.1f}s  rc={rc}")
    print(f"      manifest: {os.path.join(args.generated_dir, experiment.MANIFEST_FILENAME)}")
    print(f"      details:  {_details_path(args.generated_dir)}")
    print(f"      output:   {args.out_dir}")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
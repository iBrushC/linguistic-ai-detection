"""First OpenRouter cross-validation experiment.

Selects a deterministic subset of five authors and five essays each from
``essays.json`` (seed 1729), builds five leave-one-out folds per author,
and scores natural versus AI-recreated essays under a leakage-safe
Cosine Delta pipeline.

Outputs:

* ``generated/experiment_manifest.json`` -- the deterministic selection.
* ``src/plots/experiment_<run>/folds.json`` -- the 25 paired evaluations.
* ``src/plots/experiment_<run>/folds_cosine_delta.png`` -- bar chart.
* ``src/plots/experiment_<run>/summary.json`` -- aggregate statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import chunking  # noqa: E402
from src.distance import cosine_delta_matrix  # noqa: E402
from src.features import MFWExtractor  # noqa: E402


ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
REFERENCE_PATH = os.path.join(REPO_ROOT, "reference_corpus.json")
GENERATED_DIR = os.path.join(REPO_ROOT, "generated")
DEFAULT_MODEL = "glm-5.2"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MIN_FILL_RATIO = 0.8
DEFAULT_MFW_N = 500

MANIFEST_FILENAME = "experiment_manifest.json"
FINGERPRINT_FILENAME = "essay_details.json"
DEFAULT_RECREATION_FILENAME = "essays_glm-5.2_recreate.json"


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


# --- Selection ----------------------------------------------------------


def _token_count(text: str) -> int:
    return len(chunking.tokenize(chunking.normalize(text)))


def _author_groups(essays: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for essay in essays:
        grouped.setdefault(essay["author"], []).append(essay)
    return grouped


def select_authors_and_essays(
    essays: Sequence[dict],
    chunk_size: int,
    min_fill_ratio: float,
    min_chunks_per_essay: int,
    n_authors: int,
    n_essays_per_author: int,
    seed: int,
) -> list[dict]:
    """Deterministically pick ``n_authors`` authors and
    ``n_essays_per_author`` essays per author.

    An essay is eligible if chunking it with the configured parameters
    produces at least ``min_chunks_per_essay`` chunks.
    """
    if n_authors <= 0 or n_essays_per_author <= 0:
        raise ValueError("n_authors and n_essays_per_author must be positive")
    rng = np.random.default_rng(seed)

    eligible_by_author: dict[str, list[dict]] = {}
    for author, items in _author_groups(essays).items():
        eligible: list[dict] = []
        for essay in items:
            chunks, total = chunking.chunk_article(
                text=essay["text"],
                article_id=essay["article_id"],
                author=author,
                config=essay.get("config", ""),
                split=essay.get("split", ""),
                chunk_size=chunk_size,
                min_fill_ratio=min_fill_ratio,
            )
            if total < chunk_size * min_fill_ratio:
                continue
            if len(chunks) >= min_chunks_per_essay:
                eligible.append(essay)
        if len(eligible) >= n_essays_per_author:
            eligible_by_author[author] = eligible

    if len(eligible_by_author) < n_authors:
        raise SystemExit(
            f"only {len(eligible_by_author)} authors meet the eligibility threshold; "
            "lower --chunk-size or --min-chunks-per-essay."
        )

    author_order = sorted(eligible_by_author.keys())
    author_order.sort(key=lambda a: rng.random())
    chosen_authors = sorted(author_order[:n_authors])

    selection: list[dict] = []
    for author in chosen_authors:
        pool = eligible_by_author[author]
        order = sorted(pool, key=lambda e: rng.random())
        for essay in order[:n_essays_per_author]:
            selection.append(essay)
    selection.sort(key=lambda e: (e["author"], e["article_id"]))
    return selection


def build_manifest(
    *,
    seed: int,
    n_authors: int,
    n_essays_per_author: int,
    chunk_size: int,
    min_fill_ratio: float,
    min_chunks_per_essay: int,
    essays: Sequence[dict] | None = None,
) -> dict:
    essays = list(essays) if essays is not None else load_json(ESSAYS_PATH, default=[])
    selection = select_authors_and_essays(
        essays=essays,
        chunk_size=chunk_size,
        min_fill_ratio=min_fill_ratio,
        min_chunks_per_essay=min_chunks_per_essay,
        n_authors=n_authors,
        n_essays_per_author=n_essays_per_author,
        seed=seed,
    )
    return {
        "seed": seed,
        "n_authors": n_authors,
        "n_essays_per_author": n_essays_per_author,
        "chunk_size_tokens": chunk_size,
        "min_chunk_fill_ratio": min_fill_ratio,
        "min_chunks_per_essay": min_chunks_per_essay,
        "n_essays_total": len(selection),
        "essays": [
            {"article_id": e["article_id"], "author": e["author"], "tokens": _token_count(e["text"])}
            for e in selection
        ],
    }


def write_manifest(
    out_dir: str,
    *,
    seed: int,
    n_authors: int,
    n_essays_per_author: int,
    chunk_size: int,
    min_fill_ratio: float,
    min_chunks_per_essay: int,
) -> dict:
    manifest = build_manifest(
        seed=seed,
        n_authors=n_authors,
        n_essays_per_author=n_essays_per_author,
        chunk_size=chunk_size,
        min_fill_ratio=min_fill_ratio,
        min_chunks_per_essay=min_chunks_per_essay,
    )
    target = os.path.join(out_dir, MANIFEST_FILENAME)
    save_json_atomic(target, manifest)
    print(f"[selector] wrote {target}")
    return manifest


def load_manifest(path: str) -> dict:
    data = load_json(path, default={})
    if not data or "essays" not in data:
        raise SystemExit(f"manifest {path} is missing or malformed")
    return data


# --- Fold construction --------------------------------------------------


def build_folds(manifest: dict) -> list[dict]:
    """Five leave-one-out folds per author using the manifest essays."""
    folds: list[dict] = []
    by_author: dict[str, list[dict]] = {}
    for essay in manifest["essays"]:
        by_author.setdefault(essay["author"], []).append(essay)
    for author in sorted(by_author):
        essays = sorted(by_author[author], key=lambda e: e["article_id"])
        if len(essays) != manifest["n_essays_per_author"]:
            raise SystemExit(
                f"author {author!r} has {len(essays)} essays; "
                "manifest is inconsistent with n_essays_per_author"
            )
        for i, target in enumerate(essays):
            corpus = [e for j, e in enumerate(essays) if j != i]
            folds.append({
                "fold": len(folds),
                "author": author,
                "target_article_id": target["article_id"],
                "corpus_article_ids": [e["article_id"] for e in corpus],
                "natural_article_id": target["article_id"],
                "generated_article_id": f"{target['article_id']}::gen",
            })
    folds.sort(key=lambda f: (f["author"], f["fold"]))
    return folds


# --- Recreation loading ------------------------------------------------


def _article_record(article_id: str, body: str, *, source_article_id: str, author: str) -> dict:
    return {
        "article_id": article_id,
        "source_article_id": source_article_id,
        "author": author,
        "config": "generated",
        "split": "generated",
        "text": body,
    }


def load_natural_essays(manifest: dict) -> list[dict]:
    essays = load_json(ESSAYS_PATH, default=[])
    by_id = {e["article_id"]: e for e in essays}
    out: list[dict] = []
    for entry in manifest["essays"]:
        if entry["article_id"] not in by_id:
            raise SystemExit(f"natural essay {entry['article_id']!r} not in {ESSAYS_PATH}")
        out.append(by_id[entry["article_id"]])
    return out


def load_recreations(
    manifest: dict,
    *,
    recreation_path: str,
    model_alias: str,
) -> list[dict]:
    data = load_json(recreation_path, default=[])
    if not isinstance(data, list) or not data:
        raise SystemExit(f"no generated essays found at {recreation_path}")
    by_source: dict[str, dict] = {}
    for entry in data:
        if not entry:
            continue
        source = entry.get("source_article_id") or entry.get("article_id")
        if source:
            by_source[source] = entry
    out: list[dict] = []
    missing: list[str] = []
    for entry in manifest["essays"]:
        source = entry["article_id"]
        rec = by_source.get(source)
        if rec is None or not rec.get("body"):
            missing.append(source)
            continue
        out.append(_article_record(
            article_id=f"{source}::gen",
            body=rec["body"],
            source_article_id=source,
            author=entry["author"],
        ))
    if missing:
        raise SystemExit(
            f"missing generated essays for {missing}; rerun `python -m src.generate recreate`."
        )
    _ = model_alias
    return out


# --- Scoring -----------------------------------------------------------


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [" ".join(c["tokens"]) for c in chunks]


def fit_feature_space(
    natural_chunks: list[dict],
    reference_articles: Sequence[dict],
    mfw_n: int,
):
    """Fit MFW vocabulary on natural chunks; mu/sigma from reference only."""
    extractor = MFWExtractor(n=mfw_n)
    extractor.fit(_chunk_texts(natural_chunks), reference_corpus=[a["text"] for a in reference_articles])
    return extractor


def _fold_distance_means(
    extractor: MFWExtractor,
    natural_chunks: list[dict],
    generated_chunks: list[dict],
    corpus_chunks: list[dict],
) -> tuple[float, float]:
    X_nat = extractor.transform(_chunk_texts(natural_chunks))
    X_gen = extractor.transform(_chunk_texts(generated_chunks))
    X_corpus = extractor.transform(_chunk_texts(corpus_chunks))
    mu = np.asarray(extractor.reference_stats["mu"])
    sigma = np.asarray(extractor.reference_stats["sigma"])

    if X_corpus.shape[0] == 0:
        raise ValueError("corpus has no chunks")

    def mean_to_corpus(X_target: np.ndarray) -> float:
        if X_target.shape[0] == 0:
            return float("nan")
        X_all = np.vstack([X_target, X_corpus])
        n_t = X_target.shape[0]
        D = cosine_delta_matrix(X_all, mu, sigma)
        block = D[:n_t, n_t:]
        return float(block.mean())

    return mean_to_corpus(X_nat), mean_to_corpus(X_gen)


def score_folds(
    manifest: dict,
    folds: list[dict],
    natural_articles: list[dict],
    generated_articles: list[dict],
    reference_articles: list[dict],
    *,
    chunk_size: int,
    min_fill_ratio: float,
    mfw_n: int,
) -> list[dict]:
    natural_by_id = {a["article_id"]: a for a in natural_articles}
    generated_by_id = {a["article_id"]: a for a in generated_articles}

    selected_ids = {e["article_id"] for e in manifest["essays"]}
    selected_natural = [natural_by_id[i] for i in selected_ids if i in natural_by_id]
    natural_chunks_all = chunking.chunk_corpus(
        selected_natural,
        chunk_size=chunk_size,
        min_fill_ratio=min_fill_ratio,
    )

    extractor = fit_feature_space(natural_chunks_all, reference_articles, mfw_n=mfw_n)

    results: list[dict] = []
    for fold in folds:
        target_id = fold["target_article_id"]
        corpus_ids = fold["corpus_article_ids"]
        corpus_articles = [natural_by_id[i] for i in corpus_ids if i in natural_by_id]
        target_natural = natural_by_id[target_id]
        target_generated_id = f"{target_id}::gen"
        if target_generated_id not in generated_by_id:
            raise SystemExit(f"missing generated article for {target_id}")
        target_generated = generated_by_id[target_generated_id]

        natural_chunks = chunking.chunk_corpus(
            [target_natural],
            chunk_size=chunk_size,
            min_fill_ratio=min_fill_ratio,
        )
        generated_chunks = chunking.chunk_corpus(
            [target_generated],
            chunk_size=chunk_size,
            min_fill_ratio=min_fill_ratio,
        )
        corpus_chunks = chunking.chunk_corpus(
            corpus_articles,
            chunk_size=chunk_size,
            min_fill_ratio=min_fill_ratio,
        )

        nat_mean, gen_mean = _fold_distance_means(
            extractor, natural_chunks, generated_chunks, corpus_chunks
        )
        results.append({
            "fold": fold["fold"],
            "author": fold["author"],
            "target_article_id": target_id,
            "natural_n_chunks": len(natural_chunks),
            "generated_n_chunks": len(generated_chunks),
            "corpus_n_chunks": len(corpus_chunks),
            "natural_mean_distance": nat_mean,
            "generated_mean_distance": gen_mean,
            "delta_natural_vs_generated": float(gen_mean - nat_mean),
        })
    return results


# --- CLI ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-validate Cosine Delta on AI-recreated essays.")
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="Build the deterministic selection manifest.")
    sel.add_argument("--seed", type=int, default=1729)
    sel.add_argument("--n-authors", type=int, default=5)
    sel.add_argument("--n-essays-per-author", type=int, default=5)
    sel.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    sel.add_argument("--min-fill-ratio", type=float, default=DEFAULT_MIN_FILL_RATIO)
    sel.add_argument("--min-chunks-per-essay", type=int, default=1)
    sel.add_argument("--out-dir", default=GENERATED_DIR)

    scr = sub.add_parser("score", help="Score all folds against a manifest.")
    scr.add_argument("--manifest", required=True)
    scr.add_argument("--recreations", default=os.path.join(GENERATED_DIR, DEFAULT_RECREATION_FILENAME))
    scr.add_argument("--model", default=DEFAULT_MODEL)
    scr.add_argument("--out-dir", required=True)
    scr.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    scr.add_argument("--min-fill-ratio", type=float, default=DEFAULT_MIN_FILL_RATIO)
    scr.add_argument("--mfw-n", type=int, default=DEFAULT_MFW_N)
    return parser


def cmd_select(args) -> int:
    write_manifest(
        out_dir=args.out_dir,
        seed=args.seed,
        n_authors=args.n_authors,
        n_essays_per_author=args.n_essays_per_author,
        chunk_size=args.chunk_size,
        min_fill_ratio=args.min_fill_ratio,
        min_chunks_per_essay=args.min_chunks_per_essay,
    )
    return 0


def cmd_score(args) -> int:
    manifest = load_manifest(args.manifest)
    folds = build_folds(manifest)
    natural = load_natural_essays(manifest)
    generated = load_recreations(
        manifest, recreation_path=args.recreations, model_alias=args.model
    )
    reference = load_json(REFERENCE_PATH, default=[])
    if not reference:
        raise SystemExit(f"{REFERENCE_PATH} is missing or empty")

    results = score_folds(
        manifest,
        folds,
        natural_articles=natural,
        generated_articles=generated,
        reference_articles=reference,
        chunk_size=args.chunk_size,
        min_fill_ratio=args.min_fill_ratio,
        mfw_n=args.mfw_n,
    )

    summary = summarize_results(results)
    plot_path = write_plot(results, args.out_dir)
    folds_path = os.path.join(args.out_dir, "folds.json")
    summary_path = os.path.join(args.out_dir, "summary.json")
    save_json_atomic(folds_path, {"manifest": manifest, "folds": results})
    save_json_atomic(summary_path, summary)
    print(f"[score] wrote {folds_path}")
    print(f"[score] wrote {summary_path}")
    print(f"[score] wrote {plot_path}")
    return 0


def summarize_results(results: Sequence[dict]) -> dict:
    nat = np.array([r["natural_mean_distance"] for r in results], dtype=np.float64)
    gen = np.array([r["generated_mean_distance"] for r in results], dtype=np.float64)
    diff = gen - nat

    per_author: dict[str, dict] = {}
    for author in sorted({r["author"] for r in results}):
        mask = np.array([r["author"] == author for r in results])
        nat_a = nat[mask]
        gen_a = gen[mask]
        per_author[author] = {
            "n_folds": int(mask.sum()),
            "natural_mean": float(nat_a.mean()) if nat_a.size else float("nan"),
            "natural_std": float(nat_a.std(ddof=1)) if nat_a.size > 1 else 0.0,
            "generated_mean": float(gen_a.mean()) if gen_a.size else float("nan"),
            "generated_std": float(gen_a.std(ddof=1)) if gen_a.size > 1 else 0.0,
            "delta_mean": float((gen_a - nat_a).mean()) if nat_a.size else float("nan"),
            "delta_std": float((gen_a - nat_a).std(ddof=1)) if nat_a.size > 1 else 0.0,
        }

    return {
        "n_folds": int(len(results)),
        "overall": {
            "natural_mean": float(nat.mean()) if nat.size else float("nan"),
            "natural_std": float(nat.std(ddof=1)) if nat.size > 1 else 0.0,
            "generated_mean": float(gen.mean()) if gen.size else float("nan"),
            "generated_std": float(gen.std(ddof=1)) if gen.size > 1 else 0.0,
            "delta_mean": float(diff.mean()) if diff.size else float("nan"),
            "delta_std": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        },
        "per_author": per_author,
    }


def write_plot(results: Sequence[dict], out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    authors = sorted({r["author"] for r in results})
    nat_means: list[float] = []
    nat_sds: list[float] = []
    gen_means: list[float] = []
    gen_sds: list[float] = []
    for author in authors:
        mask = np.array([r["author"] == author for r in results])
        nat_a = np.array([r["natural_mean_distance"] for r in results])[mask]
        gen_a = np.array([r["generated_mean_distance"] for r in results])[mask]
        nat_means.append(float(nat_a.mean()))
        nat_sds.append(float(nat_a.std(ddof=1)) if nat_a.size > 1 else 0.0)
        gen_means.append(float(gen_a.mean()))
        gen_sds.append(float(gen_a.std(ddof=1)) if gen_a.size > 1 else 0.0)

    x = np.arange(len(authors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, nat_means, width, yerr=nat_sds, capsize=4,
           label="natural", color="steelblue", edgecolor="black")
    ax.bar(x + width / 2, gen_means, width, yerr=gen_sds, capsize=4,
           label="AI-recreated", color="darkorange", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(authors, rotation=30, ha="right")
    ax.set_ylabel("mean Cosine Delta to 4-essay corpus")
    ax.set_title("Cosine Delta: natural vs AI-recreated essays (5 folds per author)")
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "folds_cosine_delta.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "select":
        return cmd_select(args)
    if args.command == "score":
        return cmd_score(args)
    parser.error("unknown subcommand")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
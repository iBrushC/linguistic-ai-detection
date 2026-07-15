"""Run the full stylometric Delta pipeline end-to-end.

Loads the JSON test articles, builds a chunked corpus using
``src.chunking``, fits feature extractors on the union of test chunks and
the held-out reference corpus, computes the pairwise distance matrix
for each chosen metric, scores self-vs-cross AUC, runs a permutation
test, and writes all artifacts to ``out_dir``.

CLI:

    python -m src.run_pipeline                                  # defaults
    python -m src.run_pipeline --chunk-size 800
    python -m src.run_pipeline --distance cosine_delta --distance burrows_delta
    python -m src.run_pipeline --permutations 1000 --out-dir src/plots_full
    python -m src.run_pipeline --no-permutations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import chunking, distance, evaluation
from src.features import MFWExtractor, CharNGramExtractor
from src.topic_leakage import topic_leakage_check


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.json")


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


FEATURE_BUILDERS = {
    "mfw": lambda cfg: MFWExtractor(n=cfg["mfw_n"]),
    "char_ngram": lambda cfg: CharNGramExtractor(ngram_range=(3, 4), min_df=2),
}


DISTANCE_BUILDERS = {
    "cosine_delta": lambda X, ext: distance.cosine_delta_matrix(
        X,
        ext.reference_stats["mu"],
        ext.reference_stats["sigma"],
    ),
    "burrows_delta": lambda X, ext: distance.burrows_delta_matrix(
        X,
        ext.reference_stats["mu"],
        ext.reference_stats["sigma"],
    ),
    "char_ngram_cosine": lambda X, ext: distance.char_ngram_distance_matrix(X),
}


def _required_extractor(metric_name: str) -> str:
    return {
        "cosine_delta": "mfw",
        "burrows_delta": "mfw",
        "char_ngram_cosine": "char_ngram",
    }[metric_name]


def _chunk_dict_to_text(chunks):
    return [" ".join(c["tokens"]) for c in chunks]


def run(config: dict, out_dir: str, run_dir: str | None = None) -> dict:
    """Run the pipeline and write artifacts.

    Returns a summary dict with per-metric AUCs, p-values, and paths.
    """
    rng_seed = int(config["perm_rng_seed"])
    print(f"[run] out_dir={out_dir}")

    essays_path = os.path.join(REPO_ROOT, config["essays_path"])
    ref_path = os.path.join(REPO_ROOT, config["reference_corpus_path"])

    print(f"[run] loading test essays from {essays_path}")
    with open(essays_path, encoding="utf-8") as f:
        test_articles = json.load(f)
    print(f"[run] loading reference corpus from {ref_path}")
    with open(ref_path, encoding="utf-8") as f:
        ref_articles = json.load(f)

    print(f"[run] chunking test articles chunk_size={config['chunk_size_tokens']} "
          f"min_fill={config['min_chunk_fill_ratio']} min_chunks_per_author={config['min_chunks_per_author']}")
    chunks = chunking.chunk_corpus(
        test_articles,
        chunk_size=int(config["chunk_size_tokens"]),
        min_fill_ratio=float(config["min_chunk_fill_ratio"]),
    )
    chunking.enforce_minimum(chunks, int(config["min_chunks_per_author"]))

    report = chunking.corpus_report(chunks)
    print(f"[corpus] {report['n_chunks_total']} chunks")
    for author, info in report["per_author"].items():
        print(f"  {author:<22} chunks={info['n_chunks']:>3}  "
              f"articles={info['n_source_articles']:>3}  total_tok={info['total_chunk_tokens']:>6}")

    os.makedirs(out_dir, exist_ok=True)
    _save_json_atomic(os.path.join(out_dir, "corpus_report.json"),
                      {"config": config, **report})

    authors = [c["author"] for c in chunks]
    source_doc_ids = [c["source_doc_id"] for c in chunks]
    chunk_texts = _chunk_dict_to_text(chunks)
    ref_texts = [a["text"] for a in ref_articles]

    union_for_fitting = list(chunk_texts) + list(ref_texts)
    print(f"[run] fitting extractors on union of {len(union_for_fitting)} documents "
          f"({len(chunk_texts)} chunks + {len(ref_texts)} reference)")

    fitted = {}
    X_cache = {}
    for fname in config["features"]:
        ext = FEATURE_BUILDERS[fname](config)
        ext.fit(union_for_fitting)
        X_test = ext.transform(chunk_texts)
        fitted[fname] = ext
        X_cache[fname] = X_test

        vocab_path = os.path.join(out_dir, f"vocab_{fname}.json")
        _save_json_atomic(vocab_path, {
            "feature": fname,
            "n_features": len(ext.get_feature_names()),
            "vocab_head": ext.get_feature_names()[:50],
        })

        if ext.reference_stats is not None:
            mu = np.asarray(ext.reference_stats["mu"])
            sigma = np.asarray(ext.reference_stats["sigma"])
            zero_var = int((sigma < 1e-12).sum())
            print(f"[features] {fname}: n_features={len(ext.get_feature_names())} "
                  f"mu_nonzero={int((mu != 0).sum())} zero_sigma_cols={zero_var}")
        else:
            print(f"[features] {fname}: n_features={len(ext.get_feature_names())} "
                  "(no reference stats; tf-idf only)")

    distances_path = os.path.join(out_dir, "distances")
    os.makedirs(distances_path, exist_ok=True)

    summary: dict = {"per_metric": {}}
    for metric in config["distances"]:
        ext_name = _required_extractor(metric)
        X = X_cache[ext_name]
        ext = fitted[ext_name]

        D = DISTANCE_BUILDERS[metric](X, ext)
        np.save(os.path.join(distances_path, f"{metric}.npy"), D)

        pair_d, is_self = evaluation.pair_labels(
            D, authors, source_doc_ids,
            exclude_same_source_document=bool(config["exclude_same_source_document"]),
        )
        iu, ju, _ = evaluation._build_pair_labels(
            authors, source_doc_ids,
            exclude_same_source_document=bool(config["exclude_same_source_document"]),
        )
        observed = evaluation.auc_self_vs_cross(pair_d, is_self)
        print(f"[metric] {metric}: AUC = {observed:.4f}  "
              f"self={int(is_self.sum())} cross={int((1 - is_self).sum())}")

        if int(config.get("permutations", 0)) > 0:
            perm_summary = evaluation.permutation_auc(
                authors,
                distances=pair_d,
                is_self=is_self,
                iu=iu,
                ju=ju,
                n_permutations=int(config["permutations"]),
                rng_seed=rng_seed,
            )
        else:
            perm_summary = {
                "observed_auc": float(observed),
                "p_value": float("nan"),
                "n_permutations": 0,
                "permuted_auc_mean": float("nan"),
                "permuted_auc_50": float("nan"),
                "permuted_auc_95": float("nan"),
            }

        plot_path = os.path.join(out_dir, f"distances_overlap_{metric}.png")
        evaluation.overlap_plot(
            pair_d, is_self,
            metric_name=metric,
            save_path=plot_path,
            auc=observed,
            perm_pvalue=perm_summary.get("p_value"),
        )
        eval_payload = {
            "metric": metric,
            "feature_space": ext_name,
            "n_chunks": len(chunks),
            "n_authors": len(set(authors)),
            "n_pairs": int(len(pair_d)),
            "n_self_pairs": int(is_self.sum()),
            "n_cross_pairs": int((1 - is_self).sum()),
            "auc": observed,
            "perm": perm_summary,
            "plot_path": plot_path,
            "distance_matrix_path": os.path.join(distances_path, f"{metric}.npy"),
        }
        _save_json_atomic(os.path.join(out_dir, f"evaluation_{metric}.json"), eval_payload)
        summary["per_metric"][metric] = eval_payload

    print("\n[topic leakage] running content-vs-function-word classifier diagnostic")
    leakage = topic_leakage_check(
        [{"author": c["author"], "text": " ".join(c["tokens"])} for c in chunks],
        cv=int(config["topic_leakage_cv_folds"]),
        warning_gap=float(config["topic_leakage_warning_gap"]),
    )
    print(leakage["message"])
    _save_json_atomic(os.path.join(out_dir, "topic_leakage.json"), leakage)

    summary["config"] = config
    summary["corpus"] = report
    summary["topic_leakage"] = leakage
    _save_json_atomic(os.path.join(out_dir, "run_summary.json"), summary)

    print("\n[summary]")
    for metric, info in summary["per_metric"].items():
        p = info["perm"]["p_value"]
        p_str = f"{p:.4g}" if not (isinstance(p, float) and (p != p)) else "n/a"
        print(f"  {metric:<22} AUC = {info['auc']:.4f}  p(perm, n={info['perm']['n_permutations']}) = {p_str}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the stylometric Delta pipeline.")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON config path")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--min-chunks-per-author", type=int, default=None)
    p.add_argument("--mfw-n", type=int, default=None)
    p.add_argument("--permutations", type=int, default=None)
    p.add_argument("--no-permutations", action="store_true",
                   help="Skip the permutation test entirely.")
    p.add_argument("--feature", action="append", default=None,
                   choices=["mfw", "char_ngram"],
                   help="Restrict to a single feature extractor (repeatable).")
    p.add_argument("--distance", action="append", default=None,
                   choices=["cosine_delta", "burrows_delta", "char_ngram_cosine"],
                   help="Restrict to a single distance metric (repeatable).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (overrides config).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    if args.chunk_size is not None:
        cfg["chunk_size_tokens"] = args.chunk_size
    if args.min_chunks_per_author is not None:
        cfg["min_chunks_per_author"] = args.min_chunks_per_author
    if args.mfw_n is not None:
        cfg["mfw_n"] = args.mfw_n
    if args.permutations is not None:
        cfg["permutations"] = args.permutations
    if args.no_permutations:
        cfg["permutations"] = 0
    if args.feature is not None:
        cfg["features"] = list(args.feature)
    if args.distance is not None:
        cfg["distances"] = list(args.distance)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    elif not os.path.isabs(cfg["out_dir"]):
        cfg["out_dir"] = os.path.join(REPO_ROOT, cfg["out_dir"])

    for metric in cfg["distances"]:
        if _required_extractor(metric) not in cfg["features"]:
            raise SystemExit(
                f"distance {metric!r} requires feature {_required_extractor(metric)!r} "
                f"to be enabled in config"
            )

    run(cfg, out_dir=cfg["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

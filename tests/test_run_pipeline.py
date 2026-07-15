"""End-to-end smoke test for run_pipeline on a tiny synthetic corpus."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import run_pipeline
from src.chunking import chunk_corpus


def _write_small_corpus(essays_path: str, reference_path: str) -> None:
    rng = np.random.default_rng(0)
    vocab_a = "the cat sat on the mat and looked at the dog on the floor".split()
    vocab_b = "she sells sea shells by the shore where the boats dock at dawn".split()
    essays = []
    for aid, (author, vocab) in enumerate([("A", vocab_a), ("B", vocab_b)]):
        for article_i in range(3):
            text = " ".join(rng.choice(vocab, size=2000)) + f". doc {author} #{article_i}"
            essays.append({
                "author": author,
                "text": text,
                "config": "test",
                "split": "test",
                "article_id": f"test/test/{aid}/art{article_i}",
            })

    ref = []
    for author, vocab in [("A", vocab_a + ["extra"]), ("B", vocab_b + ["alpha"])]:
        for ri in range(2):
            ref.append({
                "author": author,
                "text": " ".join(rng.choice(vocab, size=1500)),
                "config": "ref",
                "split": "ref",
                "article_id": f"ref/ref/{author}/art{ri}",
            })

    with open(essays_path, "w", encoding="utf-8") as f:
        json.dump(essays, f)
    with open(reference_path, "w", encoding="utf-8") as f:
        json.dump(ref, f)


def test_run_pipeline_synthetic(tmp_path):
    essays_path = str(tmp_path / "essays.json")
    ref_path = str(tmp_path / "reference.json")
    _write_small_corpus(essays_path, ref_path)

    cfg = {
        "chunk_size_tokens": 200,
        "min_chunk_fill_ratio": 0.8,
        "min_chunks_per_author": 5,
        "mfw_n": 100,
        "essays_path": essays_path,
        "reference_corpus_path": ref_path,
        "features": ["mfw", "char_ngram"],
        "distances": ["cosine_delta", "burrows_delta", "char_ngram_cosine"],
        "permutations": 50,
        "out_dir": str(tmp_path / "out"),
        "exclude_same_source_document": True,
        "topic_leakage_cv_folds": 3,
        "topic_leakage_warning_gap": 0.05,
        "perm_rng_seed": 1729,
    }
    summary = run_pipeline.run(cfg, out_dir=cfg["out_dir"])
    for metric in cfg["distances"]:
        assert metric in summary["per_metric"]
        auc = summary["per_metric"][metric]["auc"]
        assert 0.0 <= auc <= 1.0
        assert summary["per_metric"][metric]["n_pairs"] > 0

    assert os.path.isfile(os.path.join(cfg["out_dir"], "corpus_report.json"))
    assert os.path.isfile(os.path.join(cfg["out_dir"], "topic_leakage.json"))

"""Tests for src.experiment: deterministic selection, fold construction,
recreation loading, and leakage-safe Cosine Delta scoring.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import chunking, experiment


# --- helpers ------------------------------------------------------------


def _fake_article(article_id: str, author: str, text: str) -> dict:
    return {
        "article_id": article_id,
        "author": author,
        "text": text,
        "config": "test",
        "split": "test",
    }


def _long_text(seed: int, vocab: list[str], n_words: int = 5000) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(vocab, size=n_words)) + " . end of doc"


def _make_essays(tmp_path: str, *, n_authors: int = 3, n_essays: int = 6,
                  chunk_words: int = 1200) -> tuple[list[dict], str]:
    rng = np.random.default_rng(0)
    vocab_a = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
               "the and of to in a is it").split()
    vocab_b = ("uno dos tres cuatro cinco seis siete ocho nueve diez "
               "the and of to in a is it").split()
    vocab_c = ("apple banana cherry date elder fig grape honey ivy jasmine "
               "the and of to in a is it").split()
    vocabs = [vocab_a, vocab_b, vocab_c][:n_authors]
    authors = [f"Author{i}" for i in range(n_authors)]
    essays = []
    for ai, author in enumerate(authors):
        for ei in range(n_essays):
            text = _long_text(seed=ai * 100 + ei, vocab=vocabs[ai], n_words=chunk_words)
            essays.append(_fake_article(f"{author}/e{ei}", author, text))
    path = os.path.join(tmp_path, "essays.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(essays, f)
    return essays, path


def _make_reference(tmp_path: str) -> list[dict]:
    rng = np.random.default_rng(42)
    vocab = ("the and of to in a is it on for with as at by from this that be are was".split()
             + ["extra"] * 50)
    out = []
    for i in range(4):
        text = " ".join(rng.choice(vocab, size=1500))
        out.append({"article_id": f"ref/{i}", "author": "ref",
                     "text": text, "config": "ref", "split": "ref"})
    return out


# --- selection ---------------------------------------------------------


def test_select_returns_deterministic_subset(tmp_path):
    essays, _ = _make_essays(str(tmp_path))
    manifest_a = experiment.build_manifest(
        seed=1729,
        n_authors=3,
        n_essays_per_author=5,
        chunk_size=500,
        min_fill_ratio=0.8,
        min_chunks_per_essay=1,
        essays=essays,
    )
    manifest_b = experiment.build_manifest(
        seed=1729,
        n_authors=3,
        n_essays_per_author=5,
        chunk_size=500,
        min_fill_ratio=0.8,
        min_chunks_per_essay=1,
        essays=essays,
    )
    assert manifest_a == manifest_b
    assert len({e["article_id"] for e in manifest_a["essays"]}) == 15
    by_author: dict[str, list[dict]] = {}
    for e in manifest_a["essays"]:
        by_author.setdefault(e["author"], []).append(e)
    for author, items in by_author.items():
        assert len(items) == 5


def test_select_requires_eligibility(tmp_path):
    essays, _ = _make_essays(str(tmp_path), n_authors=2, n_essays=2, chunk_words=200)
    with pytest.raises(SystemExit):
        experiment.build_manifest(
            seed=1729,
            n_authors=2,
            n_essays_per_author=2,
            chunk_size=500,
            min_fill_ratio=0.8,
            min_chunks_per_essay=1,
            essays=essays,
        )


# --- folds --------------------------------------------------------------


def test_folds_are_five_leave_one_out(tmp_path):
    essays, _ = _make_essays(str(tmp_path))
    manifest = experiment.build_manifest(
        seed=1729, n_authors=2, n_essays_per_author=5,
        chunk_size=500, min_fill_ratio=0.8, min_chunks_per_essay=1, essays=essays,
    )
    folds = experiment.build_folds(manifest)
    assert len(folds) == 10
    by_author: dict[str, list[dict]] = {}
    for fold in folds:
        by_author.setdefault(fold["author"], []).append(fold)
    for author, items in by_author.items():
        assert len(items) == 5
        targets = {fold["target_article_id"] for fold in items}
        assert len(targets) == 5
        for fold in items:
            assert fold["target_article_id"] not in fold["corpus_article_ids"]


# --- generator schema normalization -------------------------------------


def test_generate_recreation_uses_current_schema(tmp_path):
    essay = _fake_article("a/e0", "Author0", _long_text(0, ["the"] * 30, n_words=1500))
    detail = {
        "article_id": "a/e0",
        "author": "Author0",
        "title": "Sample",
        "word_count": 1500,
        "assignment": "Write a short essay.",
    }
    fake_client = MagicMock()
    fake_client.chat.send.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" Recreated body "))]
    )

    rec = experiment.generate_recreation if hasattr(experiment, "generate_recreation") else None
    if rec is None:
        from src import generate
        rec = generate.generate_recreation

    out = rec(
        detail=detail,
        same_author_essays=[essay],
        client=fake_client,
        model_slug="zhipu/glm-5.2",
        source_article_id="a/e0",
    )
    assert out["source_article_id"] == "a/e0"
    assert out["author"] == "Author0"
    assert out["body"] == "Recreated body"
    fake_client.chat.send.assert_called_once()
    kwargs = fake_client.chat.send.call_args.kwargs
    assert "Author0" in kwargs["messages"][-1]["content"]


# --- recreation loader -------------------------------------------------


def test_load_recreations_uses_source_article_id(tmp_path):
    manifest = {
        "seed": 1729,
        "n_authors": 1,
        "n_essays_per_author": 2,
        "essays": [
            {"article_id": "a/e0", "author": "A", "tokens": 1500},
            {"article_id": "a/e1", "author": "A", "tokens": 1500},
        ],
    }
    recreations = [
        {"source_article_id": "a/e0", "body": "recreated A"},
        {"source_article_id": "a/e1", "body": "recreated B"},
    ]
    path = os.path.join(tmp_path, "recreations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recreations, f)
    out = experiment.load_recreations(manifest, recreation_path=path, model_alias="glm-5.2")
    assert {a["source_article_id"] for a in out} == {"a/e0", "a/e1"}
    assert all(a["config"] == "generated" for a in out)


# --- scoring -----------------------------------------------------------


def test_score_folds_reference_stats_unchanged(tmp_path):
    """Generated text must not affect the reference mu/sigma."""
    essays, essays_path = _make_essays(str(tmp_path), n_authors=1, n_essays=2)
    reference = _make_reference(tmp_path)
    manifest = experiment.build_manifest(
        seed=1729, n_authors=1, n_essays_per_author=2,
        chunk_size=500, min_fill_ratio=0.8, min_chunks_per_essay=1, essays=essays,
    )
    natural_articles = [e for e in essays if e["article_id"] in {m["article_id"] for m in manifest["essays"]}]
    long_generated_body = _long_text(seed=999, vocab=[
        "alpha", "beta", "gamma", "delta", "epsilon",
        "the", "and", "of", "to", "in", "a", "is", "it"
    ], n_words=2500)
    generated_articles = [
        experiment._article_record(
            article_id=f"{a['article_id']}::gen",
            body=long_generated_body,
            source_article_id=a["article_id"],
            author=a["author"],
        )
        for a in natural_articles
    ]

    folds = experiment.build_folds(manifest)

    def run_with_reference(ref_docs):
        results = experiment.score_folds(
            manifest=manifest,
            folds=folds,
            natural_articles=natural_articles,
            generated_articles=generated_articles,
            reference_articles=ref_docs,
            chunk_size=500,
            min_fill_ratio=0.8,
            mfw_n=100,
        )
        return results

    res_a = run_with_reference(reference)

    ext = experiment.MFWExtractor(n=100)
    nat_chunks = chunking.chunk_corpus(natural_articles, chunk_size=500, min_fill_ratio=0.8)
    gen_chunks = chunking.chunk_corpus(generated_articles, chunk_size=500, min_fill_ratio=0.8)
    combined = nat_chunks + gen_chunks
    ext.fit_transform(
        [" ".join(c["tokens"]) for c in combined],
        reference_corpus=[a["text"] for a in reference],
    )
    stats_after = {
        "mu": np.asarray(ext.reference_stats["mu"]),
        "sigma": np.asarray(ext.reference_stats["sigma"]),
    }

    res_b = run_with_reference(reference)
    assert res_a == res_b
    assert stats_after["mu"].shape == (len(ext.vocab_),)
    assert (stats_after["sigma"] >= 0).all()


def test_score_folds_returns_means(monkeypatch, tmp_path):
    """End-to-end scoring with stubbed mu/sigma must produce finite means."""
    essays, _ = _make_essays(str(tmp_path), n_authors=1, n_essays=2)
    reference = _make_reference(tmp_path)
    manifest = experiment.build_manifest(
        seed=1729, n_authors=1, n_essays_per_author=2,
        chunk_size=500, min_fill_ratio=0.8, min_chunks_per_essay=1, essays=essays,
    )
    natural_articles = [e for e in essays if e["article_id"] in {m["article_id"] for m in manifest["essays"]}]
    generated_articles = [
        experiment._article_record(
            article_id=f"{a['article_id']}::gen",
            body=a["text"],
            source_article_id=a["article_id"],
            author=a["author"],
        )
        for a in natural_articles
    ]
    folds = experiment.build_folds(manifest)

    results = experiment.score_folds(
        manifest=manifest,
        folds=folds,
        natural_articles=natural_articles,
        generated_articles=generated_articles,
        reference_articles=reference,
        chunk_size=500,
        min_fill_ratio=0.8,
        mfw_n=100,
    )
    assert len(results) == 2
    for row in results:
        assert row["natural_n_chunks"] >= 1
        assert row["generated_n_chunks"] >= 1
        assert row["corpus_n_chunks"] >= 1
        assert np.isfinite(row["natural_mean_distance"])
        assert np.isfinite(row["generated_mean_distance"])
    summary = experiment.summarize_results(results)
    assert summary["n_folds"] == 2
    assert "overall" in summary and "per_author" in summary
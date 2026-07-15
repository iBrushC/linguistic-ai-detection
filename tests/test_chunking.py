"""Tests for src.chunking: boundary behaviour, drop rules, count checks."""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import chunking


def _fake_article(text: str, article_id: str = "x", author: str = "A") -> dict:
    return {
        "author": author,
        "text": text,
        "config": "test",
        "split": "test",
        "article_id": article_id,
    }


def test_tokenize_basic():
    tokens = chunking.tokenize("Hello, world! How are you?")
    assert "Hello" in tokens and "," in tokens
    assert "world" in tokens
    assert "you" in tokens


def test_normalize_collapses_whitespace():
    assert chunking.normalize("hello\n\n  world\t\tfoo ") == "hello world foo"


def test_chunk_article_basic():
    text = "the quick brown fox jumps over the lazy dog " * 500
    chunks, total = chunking.chunk_article(
        text, article_id="doc1", author="X", config="c", split="s",
        chunk_size=100, min_fill_ratio=0.8,
    )
    assert total > 0
    assert total == len(chunking.tokenize(chunking.normalize(text)))
    assert all(c["n_tokens"] == 100 for c in chunks[:-1])
    assert chunks[-1]["n_tokens"] >= 80
    assert all(c["source_doc_id"] == "doc1" for c in chunks)


def test_short_trailing_chunk_dropped():
    text = "the quick brown fox jumps over the lazy dog " * 500 + "short"
    chunks, total = chunking.chunk_article(
        text, article_id="doc2", author="X", config="c", split="s",
        chunk_size=100, min_fill_ratio=0.8,
    )
    assert all(c["n_tokens"] == 100 for c in chunks)
    assert chunks[-1]["n_tokens"] == 100


def test_short_article_no_chunks():
    chunks, total = chunking.chunk_article(
        "the quick brown fox", article_id="short", author="X",
        config="c", split="s", chunk_size=100, min_fill_ratio=0.8,
    )
    assert total < 80
    assert chunks == []
    assert total >= 0


def test_chunk_corpus_mixed():
    articles = [
        _fake_article("foo " * 500, "a1", "A"),
        _fake_article("bar " * 50, "a2", "A"),
        _fake_article("baz " * 300, "b1", "B"),
    ]
    out = chunking.chunk_corpus(articles, chunk_size=100, min_fill_ratio=0.8)
    authors = {c["author"] for c in out}
    assert authors == {"A", "B"}
    docs = {c["source_doc_id"] for c in out}
    assert "a2" not in docs


def test_enforce_minimum_fails_when_below():
    chunks = [
        {"author": "A", "source_doc_id": f"d{i}", "tokens": [], "n_tokens": 100,
         "config": "c", "split": "s", "chunk_id": f"d{i}::c000", "chunk_index": 0}
        for i in range(10)
    ]
    with pytest.raises(SystemExit):
        chunking.enforce_minimum(chunks, 15)


def test_enforce_minimum_passes_when_met():
    chunks = [
        {"author": "A", "source_doc_id": f"d{i}", "tokens": [], "n_tokens": 100,
         "config": "c", "split": "s", "chunk_id": f"d{i}::c000", "chunk_index": 0}
        for i in range(20)
    ]
    chunking.enforce_minimum(chunks, 15)  # does not raise


def test_corpus_report_keys():
    chunks = [
        {"author": "A", "source_doc_id": f"d{i}", "n_tokens": 100,
         "config": "c", "split": "s", "chunk_id": f"d{i}::c000", "chunk_index": 0,
         "tokens": []}
        for i in range(5)
    ] + [
        {"author": "B", "source_doc_id": f"e{i}", "n_tokens": 100,
         "config": "c", "split": "s", "chunk_id": f"e{i}::c000", "chunk_index": 0,
         "tokens": []}
        for i in range(7)
    ]
    rep = chunking.corpus_report(chunks)
    assert rep["n_chunks_total"] == 12
    assert rep["per_author"]["A"]["n_chunks"] == 5
    assert rep["per_author"]["A"]["n_source_articles"] == 5
    assert rep["per_author"]["B"]["n_chunks"] == 7
    assert "c/s" in rep["by_config_split"]

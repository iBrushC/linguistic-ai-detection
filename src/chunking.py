"""Chunk texts into equal-length token slabs.

A chunk is a contiguous slice of an article's token stream of
``chunk_size_tokens`` tokens. The trailing slice of any article is dropped
unless it clears ``min_chunk_fill_ratio * chunk_size_tokens``; this keeps
chunks approximately equal-sized and bounds how much overshoot / undershoot
a single low-volume article can contribute.

Tokens are produced by NLTK ``word_tokenize``; punctuation is split off so
counts are comparable across chunks.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Iterable

import nltk
from nltk.tokenize import word_tokenize


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MIN_FILL_RATIO = 0.8


def _ensure_nltk() -> None:
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def tokenize(text: str) -> list[str]:
    """Word-tokenize a text. Cached lookup of NLTK resources."""
    _ensure_nltk()
    return word_tokenize(text)


_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse runs of whitespace to a single space; strip ends."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def chunk_article(
    text: str,
    article_id: str,
    author: str,
    config: str,
    split: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
) -> tuple[list[dict], int]:
    """Slice one article into chunks.

    Returns ``(chunks, total_tokens)``. ``total_tokens`` is the length of
    the token stream *before* chunking; useful for the corpus report.
    """
    tokens = tokenize(normalize(text))
    total = len(tokens)
    chunks: list[dict] = []
    min_len = max(int(chunk_size * min_fill_ratio), 1)
    n_full = total // chunk_size
    for i in range(n_full):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append({
            "chunk_id": f"{article_id}::c{i:03d}",
            "author": author,
            "config": config,
            "split": split,
            "source_doc_id": article_id,
            "chunk_index": i,
            "tokens": tokens[start:end],
            "n_tokens": chunk_size,
        })
    rem_start = n_full * chunk_size
    rem = tokens[rem_start:]
    if len(rem) >= min_len:
        chunks.append({
            "chunk_id": f"{article_id}::c{n_full:03d}",
            "author": author,
            "config": config,
            "split": split,
            "source_doc_id": article_id,
            "chunk_index": n_full,
            "tokens": rem,
            "n_tokens": len(rem),
        })
    return chunks, total


def chunk_corpus(
    articles: Iterable[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
) -> list[dict]:
    """Carve every article into chunks and return the flat list."""
    out: list[dict] = []
    for a in articles:
        text = a.get("text") or a.get("body") or ""
        article_id = a["article_id"]
        out.extend(chunk_article(
            text=text,
            article_id=article_id,
            author=a["author"],
            config=a.get("config", ""),
            split=a.get("split", ""),
            chunk_size=chunk_size,
            min_fill_ratio=min_fill_ratio,
        )[0])
    return out


def corpus_report(chunks: list[dict]) -> dict:
    """Build a per-author and per-source-dataset chunk report."""
    by_author: dict[str, dict] = defaultdict(lambda: {
        "n_chunks": 0,
        "total_chunk_tokens": 0,
        "n_source_articles": 0,
        "min_tokens": None,
        "max_tokens": None,
    })
    by_author_articles: dict[str, set] = defaultdict(set)
    by_config_split: dict[tuple[str, str], int] = defaultdict(int)

    for c in chunks:
        a = c["author"]
        info = by_author[a]
        info["n_chunks"] += 1
        info["total_chunk_tokens"] += c["n_tokens"]
        info["min_tokens"] = c["n_tokens"] if info["min_tokens"] is None else min(info["min_tokens"], c["n_tokens"])
        info["max_tokens"] = c["n_tokens"] if info["max_tokens"] is None else max(info["max_tokens"], c["n_tokens"])
        by_author_articles[a].add(c["source_doc_id"])
        by_config_split[(c["config"], c["split"])] += 1

    out = {}
    for author, info in by_author.items():
        nc = max(info["n_chunks"], 1)
        info["mean_tokens_per_chunk"] = info["total_chunk_tokens"] / nc
        info["n_source_articles"] = len(by_author_articles[author])
        out[author] = info

    return {
        "n_chunks_total": len(chunks),
        "per_author": dict(sorted(out.items())),
        "by_config_split": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_config_split.items())},
    }


def enforce_minimum(
    chunks: list[dict],
    min_chunks_per_author: int,
    authors: Iterable[str] | None = None,
) -> None:
    """Raise ``SystemExit`` if any author falls below the floor."""
    seen_authors = set(authors) if authors is not None else None
    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        if seen_authors is not None and c["author"] not in seen_authors:
            continue
        counts[c["author"]] += 1
    bad = {a: n for a, n in counts.items() if n < min_chunks_per_author}
    if bad:
        raise SystemExit(
            f"insufficient chunks per author (min={min_chunks_per_author}): "
            f"{bad}; lower --chunk-size, raise corpus, or relax --min-chunks-per-author"
        )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MIN_FILL_RATIO",
    "tokenize",
    "normalize",
    "chunk_article",
    "chunk_corpus",
    "corpus_report",
    "enforce_minimum",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk the test corpus into equal-length slabs.")
    parser.add_argument("--chunks-out", default=os.path.join(REPO_ROOT, "src", "plots", "chunks.json"))
    parser.add_argument("--report-out", default=os.path.join(REPO_ROOT, "src", "plots", "corpus_report.json"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-fill-ratio", type=float, default=DEFAULT_MIN_FILL_RATIO)
    parser.add_argument("--min-chunks-per-author", type=int, default=15)
    parser.add_argument("--essays", default=os.path.join(REPO_ROOT, "essays.json"))
    args = parser.parse_args()

    with open(args.essays, encoding="utf-8") as f:
        articles = json.load(f)
    chunks = chunk_corpus(articles, chunk_size=args.chunk_size, min_fill_ratio=args.min_fill_ratio)
    enforce_minimum(chunks, args.min_chunks_per_author)
    serialized = [
        {k: v for k, v in c.items() if k != "tokens"} | {"token_text": " ".join(c["tokens"])}
        for c in chunks
    ]
    os.makedirs(os.path.dirname(args.chunks_out), exist_ok=True)
    with open(args.chunks_out, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    report = corpus_report(chunks)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"chunk_size": args.chunk_size, **report, "min_chunks_per_author": args.min_chunks_per_author}, indent=2))

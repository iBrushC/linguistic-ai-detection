"""Dataset loader.

Builds two disjoint slices from the `Efstathios/guardian_authorship` HF dataset:

* **test essays**:    articles for the five target authors pulled from
  ``cross_topic_1`` (all splits). These become ``essays.json`` and are
  carved into chunks for evaluation.
* **reference corpus**: articles for the same five authors pulled from
  ``cross_genre_2``, ``cross_genre_3``, ``cross_genre_4`` (all splits).
  These become ``reference_corpus.json`` and provide the z-score
  statistics for every Delta-style distance. By design the reference is
  out-of-genre so it cannot silently leak into the test set.

Each record is tagged with ``(config, split, author, article_id)`` so
article-level identity is preserved through chunking; the evaluation
layer uses the article_id to drop same-source-document pairs.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Iterable

from datasets import load_dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
REFERENCE_PATH = os.path.join(REPO_ROOT, "reference_corpus.json")

DATASET_NAME = "Efstathios/guardian_authorship"
TEST_CONFIG = "cross_topic_1"
REFERENCE_CONFIGS = ("cross_genre_2", "cross_genre_3", "cross_genre_4")
SPLITS = ("train", "validation", "test")

AUTHOR_NAMES = {
    0: "Catherine Bennett",
    1: "George Monbiot",
    2: "Hugo Young",
    3: "Jonathan Freedland",
    4: "Martin Kettle",
}


def _load_config(config: str) -> dict[str, list[dict]]:
    ds = load_dataset(DATASET_NAME, config, trust_remote_code=True)
    by_split: dict[str, list[dict]] = {}
    for split in SPLITS:
        if split in ds:
            by_split[split] = list(ds[split])
    return by_split


def _author_name(author_id: int) -> str:
    return AUTHOR_NAMES.get(int(author_id), f"author_{author_id}")


def _collect_articles(
    config: str,
    target_authors: Iterable[int],
) -> list[dict]:
    """Yield one record per article for ``target_authors`` across all splits."""
    out: list[dict] = []
    for split, rows in _load_config(config).items():
        for idx, row in enumerate(rows):
            aid = int(row["author"])
            if aid not in target_authors:
                continue
            text = (row["article"] or "").strip()
            if not text:
                continue
            row_id = row.get("id") or f"{split}_{idx}"
            out.append({
                "config": config,
                "split": split,
                "author_id": aid,
                "author": _author_name(aid),
                "article_id": f"{config}/{split}/{aid}/{row_id}",
                "topic": str(row.get("topic", "")),
                "text": text,
            })
    return out


def build_test_essays(target_authors: Iterable[int] | None = None) -> list[dict]:
    target = tuple(target_authors) if target_authors is not None else tuple(AUTHOR_NAMES)
    records = _collect_articles(TEST_CONFIG, target)
    return [{"author": r["author"], "text": r["text"],
             "config": r["config"], "split": r["split"],
             "article_id": r["article_id"]} for r in records]


def build_reference_corpus(target_authors: Iterable[int] | None = None) -> list[dict]:
    target = tuple(target_authors) if target_authors is not None else tuple(AUTHOR_NAMES)
    records: list[dict] = []
    for cfg in REFERENCE_CONFIGS:
        records.extend(_collect_articles(cfg, target))
    return [{
        "author": r["author"],
        "text": r["text"],
        "config": r["config"],
        "split": r["split"],
        "article_id": r["article_id"],
    } for r in records] 


def _summarise(records: list[dict], label: str) -> dict:
    by_author = defaultdict(list)
    for r in records:
        n = len(r["text"].split())
        by_author[r["author"]].append(n)
    per_author = {}
    for author, lens in sorted(by_author.items()):
        per_author[author] = {
            "n_articles": len(lens),
            "total_words": sum(lens),
            "mean_words_per_article": sum(lens) / len(lens) if lens else 0.0,
            "min_words": min(lens) if lens else 0,
            "max_words": max(lens) if lens else 0,
        }
    cfg_counts = Counter((r["config"], r["split"]) for r in records)
    print(f"\n[{label}] {len(records)} records, {len(per_author)} authors")
    print(f"  totals: articles={len(records)} words={sum(sum(lens) for lens in by_author.values())}")
    for author, info in per_author.items():
        print(f"  {author:<22} articles={info['n_articles']:>3} "
              f"total_words={info['total_words']:>6} mean={info['mean_words_per_article']:>7.1f}")
    by_config: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (cfg, sp), n in cfg_counts.items():
        by_config[cfg][sp] = n
    print(f"  by (config, split): {dict({k: dict(v) for k, v in by_config.items()})}")
    return {"per_author": per_author, "by_config_split": {k: dict(v) for k, v in by_config.items()}}


def main() -> None:
    test_records = build_test_essays()
    summary = _summarise(test_records, "test")
    with open(ESSAYS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
    print(f"  wrote {ESSAYS_PATH}")

    ref_records = build_reference_corpus()
    ref_summary = _summarise(ref_records, "reference")
    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(ref_records, f, ensure_ascii=False, indent=2)
    print(f"  wrote {REFERENCE_PATH}")

    full_report = {
        "test": summary,
        "reference": ref_summary,
        "target_authors": AUTHOR_NAMES,
        "test_config": TEST_CONFIG,
        "reference_configs": list(REFERENCE_CONFIGS),
    }
    report_path = os.path.join(REPO_ROOT, "src", "plots", "corpus_report_raw.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()

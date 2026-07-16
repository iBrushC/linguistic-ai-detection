"""Smoke test for ``scripts/final_analysis.py``.

Constructs minimal synthetic folds / recreations / essay-details under a tmp
root, invokes the real entrypoint, and asserts every expected artifact
lands on disk with non-zero size.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


EXPECTED_CSVS = [
    "ranking_cosine_delta.csv",
    "ranking_cosine_match.csv",
    "author_breakdown.csv",
    "trick_analysis.csv",
    "cost_vs_performance.csv",
    "detection_roc.csv",
    "per_author_heatmap.csv",
    "model_agreement_jaccard.csv",
]
EXPECTED_PNGS = [
    "ranking_cosine_delta.png",
    "ranking_cosine_match.png",
    "author_breakdown.png",
    "trick_analysis.png",
    "cost_vs_performance.png",
    "detection_roc.png",
    "per_author_heatmap.png",
    "model_agreement.png",
    "jaccard_overlap.png",
    "delta_distribution.png",
]
REQUIRED_ARTIFACTS = EXPECTED_CSVS + EXPECTED_PNGS + ["final_report.md"]


def _write_minimal_dataset(tmp_path, models_all: dict[str, str]):
    """Lay down minimal essays.json, essay_details.json, models.json, plus
    one essays_<alias>_recreate.json and one experiment_<alias>/{folds,summary}.json
    per alias, all synthetic."""
    # 5 authors, 5 essays each -- mimics the manifest layout.
    authors = [
        "Catherine Bennett",
        "George Monbiot",
        "Hugo Young",
        "Jonathan Freedland",
        "Martin Kettle",
    ]
    essays = []
    for ai, author in enumerate(authors):
        for ei in range(5):
            essays.append({
                "article_id": f"split{ai}/essay{ei}",
                "author": author,
                "title": f"essay{ei}",
                "text": ("Plain essay body written for testing the final "
                         "analysis smoke test. " * 80),
                "config": f"split{ai}",
                "split": "test",
                "source_doc_id": f"split{ai}/essay{ei}",
            })
    (tmp_path / "essays.json").write_text(json.dumps(essays), encoding="utf-8")

    # essay_details.json -- one assignment per essay (only the 5x5=25 manifest essays).
    manifest_essays = [
        {"article_id": f"split{ai}/essay{ei}", "author": authors[ai]}
        for ai in range(5) for ei in range(5)
    ]
    details = []
    for e in manifest_essays:
        details.append({
            "article_id": e["article_id"],
            "author": e["author"],
            "title": e["article_id"],
            "word_count": 1000,
            "assignment": "Write a test essay that imitates the author.",
            "model_slug": "test/test",
            "model_alias": "glm-5.2",
        })
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "essay_details.json").write_text(json.dumps(details), encoding="utf-8")

    # experiment_manifest.json -- shared across all aliases.
    manifest = {
        "seed": 1729,
        "n_authors": 5,
        "n_essays_per_author": 5,
        "chunk_size_tokens": 1000,
        "min_chunk_fill_ratio": 0.8,
        "min_chunks_per_essay": 1,
        "n_essays_total": 25,
        "essays": [{"article_id": e["article_id"], "author": e["author"], "tokens": 1500}
                   for e in manifest_essays],
    }
    (generated / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # models.json
    (tmp_path / "models.json").write_text(json.dumps(models_all), encoding="utf-8")

    # For each alias, write minimal recreations and a synthetic folds/summary
    # that produce ALL delta < 0 (so the trick-rate is 100% for predictability).
    out_root = tmp_path / "plots"
    for alias in models_all:
        recs = []
        for e in manifest_essays:
            recs.append({
                "article_id": e["article_id"],
                "source_article_id": e["article_id"],
                "author": e["author"],
                "title": e["article_id"],
                "body": ("body " * 900).strip(),  # ~900 words => ~1000 tokens
                "model_slug": alias,
                "model_alias": alias,
                "word_count": 900,
            })
        (generated / f"essays_{alias}_recreate.json").write_text(
            json.dumps(recs), encoding="utf-8"
        )

        folds = []
        for fi, e in enumerate(manifest_essays):
            natural = 0.7 + (fi % 5) * 0.05  # spread natural across [0.7, 0.9]
            generated_d = natural - 0.05      # always less than natural => trick
            folds.append({
                "fold": fi,
                "author": e["author"],
                "target_article_id": e["article_id"],
                "natural_n_chunks": 1,
                "generated_n_chunks": 1,
                "corpus_n_chunks": 4,
                "natural_mean_distance": natural,
                "generated_mean_distance": generated_d,
                "delta_natural_vs_generated": generated_d - natural,
            })
        alias_dir = out_root / f"experiment_{alias}"
        alias_dir.mkdir(parents=True)
        (alias_dir / "folds.json").write_text(json.dumps(
            {"manifest": manifest, "folds": folds}, indent=2
        ), encoding="utf-8")

        # Synthesize per-author and overall stats from the folds.
        per_author: dict[str, dict] = {}
        from collections import defaultdict
        grouped = defaultdict(list)
        for f in folds:
            grouped[f["author"]].append(f)
        for author, group in sorted(grouped.items()):
            nat = [g["natural_mean_distance"] for g in group]
            gen = [g["generated_mean_distance"] for g in group]
            per_author[author] = {
                "n_folds": len(group),
                "natural_mean": sum(nat) / len(nat),
                "natural_std": 0.0,
                "generated_mean": sum(gen) / len(gen),
                "generated_std": 0.0,
                "delta_mean": sum(g["delta_natural_vs_generated"] for g in group) / len(group),
                "delta_std": 0.0,
            }
        # Overall = mean across all folds.
        all_nat = [f["natural_mean_distance"] for f in folds]
        all_gen = [f["generated_mean_distance"] for f in folds]
        all_delta = [f["delta_natural_vs_generated"] for f in folds]
        summary = {
            "n_folds": len(folds),
            "overall": {
                "natural_mean": sum(all_nat) / len(all_nat),
                "natural_std": 0.0,
                "generated_mean": sum(all_gen) / len(all_gen),
                "generated_std": 0.0,
                "delta_mean": sum(all_delta) / len(all_delta),
                "delta_std": 0.0,
            },
            "per_author": per_author,
        }
        (alias_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")

    # The cross-model aggregator has to be re-built by the driver -- but here
    # we just hand-construct the same shape so the script can read it.
    multi_dir = out_root / "experiment_multi"
    multi_dir.mkdir(parents=True)
    cross: dict = {"overall_delta_mean": {}, "per_author_delta_mean": {}}
    per_model_block: dict = {}
    for alias in models_all:
        s = json.loads((out_root / f"experiment_{alias}" / "summary.json").read_text())
        per_model_block[alias] = s
        cross["overall_delta_mean"][alias] = s["overall"]["delta_mean"]
        cross["per_author_delta_mean"][alias] = {
            a: v["delta_mean"] for a, v in s["per_author"].items()
        }
    multi_summary = {"aliases": sorted(models_all), "per_model": per_model_block, "cross": cross}
    (multi_dir / "summary.json").write_text(json.dumps(multi_summary, indent=2),
                                            encoding="utf-8")


@pytest.fixture(scope="module")
def sample_models() -> dict[str, str]:
    return json.loads(
        open(os.path.join(REPO_ROOT, "models.json"), encoding="utf-8").read()
    )


def test_final_analysis_smoke(tmp_path, sample_models, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_minimal_dataset(tmp_path, sample_models)

    out_root = tmp_path / "plots"
    rc = subprocess.check_call([
        sys.executable, os.path.join(SCRIPTS_DIR, "final_analysis.py"),
        "--out-root", str(out_root),
        "--generated-dir", str(tmp_path / "generated"),
        "--essays", str(tmp_path / "essays.json"),
        "--details", str(tmp_path / "generated" / "essay_details.json"),
        "--models-json", str(tmp_path / "models.json"),
    ])
    assert rc == 0

    out_dir = out_root / "experiment_multi" / "final"
    assert out_dir.is_dir(), f"missing output dir {out_dir}"
    for name in REQUIRED_ARTIFACTS:
        p = out_dir / name
        assert p.exists(), f"missing artifact {p}"
        assert p.stat().st_size > 0, f"empty artifact {p}"

    # Sanity-check the ranking CSV includes both natural and model bars.
    ranking = (out_dir / "ranking_cosine_delta.csv").read_text(encoding="utf-8")
    assert "Author (natural" in ranking
    for alias in sample_models:
        assert alias in ranking

    # Trick CSV must mention every alias with a 100% rate (we built all-trick folds).
    trick = (out_dir / "trick_analysis.csv").read_text(encoding="utf-8")
    assert "ALL_MODELS_POOLED" in trick

    # Cost CSV must contain every alias.
    cost = (out_dir / "cost_vs_performance.csv").read_text(encoding="utf-8")
    for alias in sample_models:
        assert alias in cost

    # Report must include the headline and caveat markers.
    report = (out_dir / "final_report.md").read_text(encoding="utf-8")
    assert "Headline numbers" in report
    assert "Pricing assumptions" in report or "Cost assumptions" in report
    assert "![" in report  # at least one embedded image


def test_final_analysis_handles_missing_fold_for_chatgpt(tmp_path, sample_models, monkeypatch):
    """If the chatgpt folds file has 1 fold with generated_n_chunks=0 (as in
    the real dataset), the script must drop it without crashing."""
    monkeypatch.chdir(tmp_path)
    _write_minimal_dataset(tmp_path, sample_models)

    # Patch the chatgpt summary to inject 0-chunk fold and disable its overall.
    folds_path = tmp_path / "plots" / "experiment_chatgpt-5.6" / "folds.json"
    folds_doc = json.loads(folds_path.read_text(encoding="utf-8"))
    folds_doc["folds"][3]["generated_n_chunks"] = 0
    folds_doc["folds"][3]["generated_mean_distance"] = None
    folds_doc["folds"][3]["delta_natural_vs_generated"] = None
    folds_path.write_text(json.dumps(folds_doc), encoding="utf-8")

    summary_path = tmp_path / "plots" / "experiment_chatgpt-5.6" / "summary.json"
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    s["overall"]["generated_mean"] = float("nan")
    s["overall"]["generated_std"] = float("nan")
    s["overall"]["delta_mean"] = float("nan")
    s["per_author"]["Catherine Bennett"]["generated_mean"] = float("nan")
    summary_path.write_text(json.dumps(s), encoding="utf-8")

    out_root = tmp_path / "plots"
    rc = subprocess.check_call([
        sys.executable, os.path.join(SCRIPTS_DIR, "final_analysis.py"),
        "--out-root", str(out_root),
    ])
    assert rc == 0
    out_dir = out_root / "experiment_multi" / "final"
    assert (out_dir / "final_report.md").exists()
    assert (out_dir / "ranking_cosine_delta.csv").exists()
    report = (out_dir / "final_report.md").read_text(encoding="utf-8")
    assert "chatgpt-5.6" in report


@pytest.fixture(scope="module", autouse=True)
def _cleanup():
    yield
    # Nothing to clean -- subprocess uses tmp_path which pytest cleans up.

"""Tests for the multi-model driver in scripts/run_multi_model_test.py."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_multi_model_test as rmm  # noqa: E402


def _copy_essays(tmp_path):
    src = os.path.join(REPO_ROOT, "essays.json")
    dst = tmp_path / "essays.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(dst)


def _read_manifest(tmp_path) -> dict:
    path = tmp_path / "generated" / "experiment_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_pipeline_completes_all_models(tmp_path, monkeypatch, capsys):
    """Offline end-to-end: all four models.json aliases produce complete artifacts."""
    _copy_essays(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    out_root = tmp_path / "plots"
    models_all = json.loads(open(os.path.join(REPO_ROOT, "models.json"), encoding="utf-8").read())
    expected_aliases = sorted(models_all)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_multi_model_test",
        "--dry-run",
        "--generated-dir", str(generated),
        "--out-root", str(out_root),
    ])

    rc = rmm.main()
    assert rc == 0

    manifest = _read_manifest(tmp_path)
    assert manifest["seed"] == 1729
    assert len(manifest["essays"]) == 25
    target_ids = {e["article_id"] for e in manifest["essays"]}

    details = json.loads((generated / "essay_details.json").read_text(encoding="utf-8"))
    have_details = {item["article_id"] for item in details if item}
    assert target_ids.issubset(have_details)

    for alias in expected_aliases:
        rec_path = generated / f"essays_{alias}_recreate.json"
        assert rec_path.exists(), f"missing recreations for {alias}"
        recs = json.loads(rec_path.read_text(encoding="utf-8"))
        rec_sources = {item["source_article_id"] for item in recs if item}
        assert target_ids.issubset(rec_sources)

        plot_dir = out_root / f"experiment_{alias}"
        assert (plot_dir / "folds.json").exists()
        assert (plot_dir / "summary.json").exists()
        assert (plot_dir / "folds_cosine_delta.png").exists()

    multi = out_root / "experiment_multi" / "summary.json"
    assert multi.exists()
    multi_data = json.loads(multi.read_text(encoding="utf-8"))
    assert set(multi_data["aliases"]) == set(expected_aliases)

    captured = capsys.readouterr().out
    assert "[plan]" in captured
    assert all(alias in captured for alias in expected_aliases)


def test_existing_completed_model_is_skipped(tmp_path, monkeypatch):
    """Models with complete recreation files must be skipped without re-generation."""
    _copy_essays(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    out_root = tmp_path / "plots"

    existing_alias = "glm-5.2"
    recreations = json.loads(
        open(
            os.path.join(REPO_ROOT, "generated", f"essays_{existing_alias}_recreate.json"),
            encoding="utf-8",
        ).read()
    )
    recreations = [r for r in recreations if r]
    sources = {r["source_article_id"] for r in recreations}
    target_path = generated / f"essays_{existing_alias}_recreate.json"
    target_path.write_text(json.dumps(recreations), encoding="utf-8")

    manifest = rmm.ensure_manifest(_args(tmp_path, generated, out_root))
    target_ids = {e["article_id"] for e in manifest["essays"]}
    assert target_ids.issubset(sources)

    monkeypatch.setattr(
        rmm,
        "process_model",
        lambda alias, args, manifest: {
            "alias": alias,
            "ok": True,
            "recreated": False,
            "scored": True,
            "log": [f"skipped {alias}"],
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "run_multi_model_test",
        "--dry-run",
        "--generated-dir", str(generated),
        "--out-root", str(out_root),
    ])

    rc = rmm.main()
    assert rc == 0


def test_force_recreates_existing_model(tmp_path, monkeypatch):
    """--force should bypass the skip-completed check for recreations."""
    _copy_essays(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    out_root = tmp_path / "plots"
    monkeypatch.chdir(tmp_path)

    created: list[str] = []
    real_process = rmm.process_model

    def spy(alias, args, manifest):
        created.append(alias)
        return real_process(alias, args, manifest)

    monkeypatch.setattr(rmm, "process_model", spy)
    monkeypatch.setattr(sys, "argv", [
        "run_multi_model_test",
        "--dry-run",
        "--force",
        "--generated-dir", str(generated),
        "--out-root", str(out_root),
    ])

    rc = rmm.main()
    assert rc == 0
    models_all = list(json.loads(open(os.path.join(REPO_ROOT, "models.json"), encoding="utf-8").read()))
    assert sorted(created) == sorted(models_all)

    target_alias = models_all[0]
    rec_path = generated / f"essays_{target_alias}_recreate.json"
    data = json.loads(rec_path.read_text(encoding="utf-8"))
    assert all(item["model_alias"] == target_alias for item in data if item)


def test_resolve_aliases_filters_missing_and_skipped():
    models_all = {"glm-5.2": "x", "chatgpt-5.6": "y", "claude-opus-4.8": "z"}
    args = type("A", (), {"models": ["glm-5.2", "bogus"], "skip_models": ["glm-5.2"]})()
    assert rmm.resolve_aliases(args, models_all) == []


def test_resolve_aliases_defaults_to_all():
    models_all = {"glm-5.2": "x", "chatgpt-5.6": "y"}
    args = type("A", (), {"models": None, "skip_models": []})()
    assert rmm.resolve_aliases(args, models_all) == ["chatgpt-5.6", "glm-5.2"]


def _args(tmp_path, generated, out_root):
    return type(
        "A",
        (),
        {
            "seed": 1729,
            "chunk_size": 1000,
            "min_fill_ratio": 0.8,
            "min_chunks_per_essay": 1,
            "generated_dir": str(generated),
            "no_select": False,
            "out_root": str(out_root),
            "dry_run": True,
            "models_concurrency": 1,
            "model_workers": 1,
            "force": False,
            "no_score": False,
            "no_summarize": False,
            "mfw_n": 500,
            "models": None,
            "skip_models": [],
        },
    )()

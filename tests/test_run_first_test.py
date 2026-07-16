"""Tests for the run_first_test.py master driver."""

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

import run_first_test  # noqa: E402


def _copy_essays(tmp_path):
    src = os.path.join(REPO_ROOT, "essays.json")
    dst = tmp_path / "essays.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(dst)


def test_dry_run_pipeline_completes(tmp_path, monkeypatch, capsys):
    """The master script must run end-to-end offline."""
    generated = tmp_path / "generated"
    generated.mkdir()
    out_dir = tmp_path / "plots"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", [
        "run_first_test",
        "--dry-run",
        "--generated-dir", str(generated),
        "--out-dir", str(out_dir),
    ])

    rc = run_first_test.main()
    assert rc == 0

    manifest_path = generated / "experiment_manifest.json"
    details_path = generated / "essay_details.json"
    recreations_path = generated / "essays_dry-run_recreate.json"
    assert manifest_path.exists()
    assert details_path.exists()
    assert recreations_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 1729
    assert manifest["n_authors"] == 5
    assert manifest["n_essays_per_author"] == 5
    assert len(manifest["essays"]) == 25

    summary_path = out_dir / "summary.json"
    folds_path = out_dir / "folds.json"
    chart_path = out_dir / "folds_cosine_delta.png"
    assert summary_path.exists()
    assert folds_path.exists()
    assert chart_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_folds"] == 25
    assert set(summary["per_author"].keys()) == {a["author"] for a in manifest["essays"]}

    captured = capsys.readouterr().out
    assert "elapsed" in captured
    assert "[1/4]" in captured and "[4/4]" in captured


def test_dry_run_idempotent_reuse(tmp_path, monkeypatch):
    """Re-running the dry-run driver must reuse existing artifacts without
    failing the network stages."""
    generated = tmp_path / "generated"
    generated.mkdir()
    out_dir = tmp_path / "plots"

    monkeypatch.setattr(sys, "argv", [
        "run_first_test",
        "--dry-run",
        "--generated-dir", str(generated),
        "--out-dir", str(out_dir),
    ])
    rc1 = run_first_test.main()
    assert rc1 == 0

    monkeypatch.setattr(sys, "argv", [
        "run_first_test",
        "--dry-run",
        "--generated-dir", str(generated),
        "--out-dir", str(out_dir),
    ])
    rc2 = run_first_test.main()
    assert rc2 == 0
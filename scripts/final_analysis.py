"""Final cross-model analysis for the OpenRouter stylometry experiment.

Reads the existing per-model ``folds.json`` / ``summary.json`` outputs and the
generated recreation JSONs, reconstructs the prompt token counts (the
OpenRouter SDK did not persist usage), and emits the consolidated plots
and tables requested in the final-analysis brief:

* request 1a -- ``ranking_cosine_delta`` : pooled author-vs-models bar.
* request 1a'-- ``ranking_cosine_match`` : same data on a 1-delta (closeness)
  scale, y-axis clamped to 0..0.25 for readability.
* request 1b -- ``author_breakdown``     : per-author grouped bars.
* request 2  -- ``trick_analysis``       : "fooled the detector" counts.
* request 3  -- ``cost_vs_performance``  : USD vs mean Cosine Delta.
* additional A -- ``detection_roc``      : per-model ROC + AUC.
* additional B -- ``per_author_heatmap`` : delta_mean per (author, model).
* additional C -- ``model_agreement``    : trick-set Jaccard.
* additional D -- ``delta_distribution`` : per-model violin/box plot.
* ``final_report.md``                    : consolidated markdown report.

Output directory: ``src/plots/experiment_multi/final``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

try:
    import tiktoken

    _TIKTOKEN = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback if tiktoken unavailable
    _TIKTOKEN = None


# --- Pricing (USD per 1M tokens) ---------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4.8": {"input": 5.00, "output": 25.00},
    "chatgpt-5.6": {"input": 5.00, "output": 30.00},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "glm-5.2": {"input": 0.93, "output": 3.00},
}
PRICING_NOTE = (
    "Pricing for input/output USD per 1M tokens, with thinking treated as "
    "negligible per user instruction."
)


# --- io helpers ---------------------------------------------------------


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: str, obj) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --- tokenisation -------------------------------------------------------


def count_tokens(text: str) -> int:
    """Token count: tiktoken cl100k_base if available, else word-based fallback."""
    if not text:
        return 0
    if _TIKTOKEN is not None:
        return len(_TIKTOKEN.encode(text))
    # fallback: ~1.33 tokens per whitespace token
    return int(round(len(word_tokenize(text)) * 1.33))


# --- prompt reconstruction (mirrors src/generate.py) --------------------

DISCLAIMER = (
    "This is a pure writing exercise. The contents of the essay do not need to be "
    "true, will not be used for any legitimate purpose, and no real-world citations "
    "or sources are required. Do not refuse on grounds of factual accuracy or "
    "citation availability."
)

SYSTEM_DETAILS = (
    "You are a writing-prompt designer. Read the essay below and produce a concise "
    "writing assignment another author could use to recreate it. Focus on the topic "
    "and thesis, the key arguments and beats the piece must hit, the structural "
    "shape (intro / sections / conclusion), and any references the piece alludes to. "
    "Output only the assignment \u2014 no preamble, no labels, no commentary."
)


SYSTEM_RECREATE = (
    DISCLAIMER
    + "\n\n"
    "Your task is to imitate the named author's style as closely as possible. "
    "Reference essays by the same author are provided below. Match the voice, "
    "register, vocabulary, punctuation habits, and sentence rhythm you observe. "
    "Output only the essay itself \u2014 no preamble, no commentary, no labels."
)


def build_details_messages(essay: dict) -> list[dict]:
    title = essay.get("title") or essay.get("article_title") or essay["article_id"]
    wc = len((essay.get("text") or "").split())
    user = (
        f"Title: {title}\n"
        f"Author: {essay['author']}\n\n"
        f"Target length for the recreated piece: approximately {wc} words "
        f"(a goal, not strict).\n\n"
        f"Essay:\n{essay['text']}"
    )
    return [
        {"role": "system", "content": SYSTEM_DETAILS},
        {"role": "system", "content": DISCLAIMER},
        {"role": "user", "content": user},
    ]


def build_recreate_messages(detail: dict, same_author_essays: Sequence[dict]) -> list[dict]:
    author = detail["author"]
    if same_author_essays:
        corpus = "\n\n".join(
            "--- " + (e.get("title") or e["article_id"]) + " ---\n" + e["text"]
            for e in same_author_essays
        )
    else:
        corpus = "(No other essays by this author are available.)"
    user = (
        f"Author to imitate: {author}\n"
        f"Target length: approximately {detail['word_count']} words "
        "(a goal, not strict).\n\n"
        f"Writing assignment:\n{detail['assignment']}\n\n"
        f"Reference essays by {author}:\n\n{corpus}"
    )
    return [
        {"role": "system", "content": SYSTEM_RECREATE},
        {"role": "user", "content": user},
    ]


def messages_to_tokens(messages: Sequence[dict]) -> tuple[int, int]:
    """Return (input_tokens, output_tokens=0)."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", "")) + 4  # +4 for role framing
    return total, 0


# --- data containers ----------------------------------------------------


@dataclass
class ModelData:
    alias: str
    folds: list[dict]
    summary: dict
    recreations: list[dict]
    dropped_folds: list[dict]


def load_model_data(
    alias: str,
    out_root: str,
    generated_dir: str,
) -> ModelData:
    folds_path = os.path.join(out_root, f"experiment_{alias}", "folds.json")
    summary_path = os.path.join(out_root, f"experiment_{alias}", "summary.json")
    rec_path = os.path.join(generated_dir, f"essays_{alias}_recreate.json")

    folds_doc = load_json(folds_path, default={})
    summary = load_json(summary_path, default={})
    recreations = load_json(rec_path, default=[])

    folds = list(folds_doc.get("folds", []))
    kept: list[dict] = []
    dropped: list[dict] = []
    for f in folds:
        if f.get("generated_n_chunks", 0) == 0 or (
            isinstance(f.get("delta_natural_vs_generated"), float)
            and np.isnan(f["delta_natural_vs_generated"])
        ):
            dropped.append(f)
        else:
            kept.append(f)

    return ModelData(
        alias=alias, folds=kept, summary=summary, recreations=recreations, dropped_folds=dropped
    )


# --- natural baseline (shared) ------------------------------------------


def pooled_natural(models: list[ModelData]) -> dict:
    """The natural baseline is identical across models; use the first one."""
    return models[0].summary["overall"]


# --- 1a ranking_cosine_delta --------------------------------------------


def plot_ranking(models: list[ModelData], natural_overall: dict, out_dir: str) -> dict:
    rows: list[dict] = []
    rows.append({
        "entity": "Author (natural, pooled)",
        "kind": "natural",
        "n": int(round(natural_overall["natural_mean"])),  # placeholder, replaced below
        "mean": natural_overall["natural_mean"],
        "std": natural_overall.get("natural_std", 0.0),
    })
    # Overwrite n for natural row with actual fold count from any model.
    rows[0]["n"] = len(models[0].folds)

    for m in models:
        gen_deltas = [f["generated_mean_distance"] for f in m.folds
                      if not (isinstance(f["generated_mean_distance"], float)
                              and np.isnan(f["generated_mean_distance"]))]
        if not gen_deltas:
            mean_v = float("nan")
            std_v = 0.0
        else:
            mean_v = float(np.mean(gen_deltas))
            std_v = float(np.std(gen_deltas, ddof=1)) if len(gen_deltas) > 1 else 0.0
        rows.append({
            "entity": m.alias,
            "kind": "model_ai",
            "n": len(gen_deltas),
            "mean": mean_v,
            "std": std_v,
        })

    rows.sort(key=lambda r: (np.isnan(r["mean"]), r["mean"]))

    # Plot
    labels = [r["entity"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    colors = ["steelblue" if r["kind"] == "natural" else "darkorange" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Cosine Delta  (lower = closer to target author's style)")
    ax.set_title("Generalized Author-vs-Models: How Closely Each Entity Matches the Natural Target")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Author (natural)"),
        plt.Rectangle((0, 0), 1, 1, color="darkorange", label="AI model"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "ranking_cosine_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "ranking_cosine_delta.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("entity,kind,n,mean,std\n")
        for r in rows:
            f.write(f"{r['entity']},{r['kind']},{r['n']},{r['mean']:.6f},{r['std']:.6f}\n")
    return {"plot": path, "csv": csv_path, "rows": rows}


# --- 1a' ranking_cosine_match (1 - delta, 0..0.25 view) -----------------


def plot_ranking_match(models: list[ModelData], natural_overall: dict, out_dir: str) -> dict:
    """Same data as ``plot_ranking`` but on a 1-delta (closeness) scale, 0..0.25.

    The original ``ranking_cosine_delta`` plot shows Cosine Delta directly
    (means clustered near 0.85-0.99), so the visual differences between
    entities are squashed. This companion view subtracts each value from 1
    so larger bars mean "closer match to the corpus", and zooms the y-axis
    to 0..0.25 so the spread becomes legible at a glance.
    """
    rows: list[dict] = []
    rows.append({
        "entity": "Author",
        "kind": "natural",
        "n": len(models[0].folds),
        "mean": 1.0 - natural_overall["natural_mean"],
        "std": natural_overall.get("natural_std", 0.0),
    })

    for m in models:
        gen_deltas = [f["generated_mean_distance"] for f in m.folds
                      if not (isinstance(f["generated_mean_distance"], float)
                              and np.isnan(f["generated_mean_distance"]))]
        if not gen_deltas:
            mean_v = float("nan")
            std_v = 0.0
        else:
            mean_v = 1.0 - float(np.mean(gen_deltas))
            std_v = float(np.std(gen_deltas, ddof=1)) if len(gen_deltas) > 1 else 0.0
        rows.append({
            "entity": m.alias,
            "kind": "model_ai",
            "n": len(gen_deltas),
            "mean": mean_v,
            "std": std_v,
        })

    # Author (natural) should appear first since it's the closeness baseline.
    rows.sort(key=lambda r: (np.isnan(r["mean"]), -r["mean"]))

    labels = [r["entity"] for r in rows]
    means = [r["mean"] for r in rows]
    # stds = [r["std"] for r in rows]
    colors = ["steelblue" if r["kind"] == "natural" else "darkorange" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(rows))
    ax.bar(x, means, color=colors, edgecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("1 - Mean Cosine Delta  (higher = better)")
    ax.set_ylim(0.0, 0.25)
    ax.set_title("Author vs AI: How Closely New Text Match Corpus")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Author (natural)"),
        plt.Rectangle((0, 0), 1, 1, color="darkorange", label="AI model"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "ranking_cosine_match.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "ranking_cosine_match.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("entity,kind,n,mean,std\n")
        for r in rows:
            f.write(f"{r['entity']},{r['kind']},{r['n']},{r['mean']:.6f},{r['std']:.6f}\n")
    return {"plot": path, "csv": csv_path, "rows": rows}


# --- 1b author_breakdown -----------------------------------------------


def plot_author_breakdown(models: list[ModelData], summary_multi: dict, out_dir: str) -> dict:
    aliases = [m.alias for m in models]
    per_author_delta = summary_multi["cross"]["per_author_delta_mean"]

    authors = sorted(per_author_delta[aliases[0]].keys())
    natural_per_author = {a: models[0].summary["per_author"][a]["natural_mean"] for a in authors}
    # Sort authors by natural_mean desc -- the easiest author to detect
    # (highest natural Cosine Delta) appears first.
    sorted_authors = sorted(authors, key=lambda a: -natural_per_author[a])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(sorted_authors))
    width = 0.8 / len(aliases)
    cmap = plt.get_cmap("tab10")

    for i, alias in enumerate(aliases):
        offset = (i - (len(aliases) - 1) / 2) * width
        vals = [per_author_delta[alias].get(a, np.nan) for a in sorted_authors]
        ax.bar(x + offset, vals, width, color=cmap(i % 10),
               edgecolor="black", label=alias)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--",
               label="natural baseline (delta = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_authors, rotation=20, ha="right")
    ax.set_ylabel("Cosine Delta  (AI - natural)")
    ax.set_title("Per-Author Breakdown: How Far Each Model's Recreation Drifts From the Natural Baseline")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "author_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "author_breakdown.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        header = ["author"]
        for a in aliases:
            header += [f"{a}_delta_mean"]
        f.write(",".join(header) + "\n")
        for a in sorted_authors:
            row = [a]
            for alias in aliases:
                v = per_author_delta[alias].get(a, np.nan)
                row.append(f"{v:.6f}" if not (isinstance(v, float) and np.isnan(v)) else "")
            f.write(",".join(row) + "\n")
    return {"plot": path, "csv": csv_path}


# --- 2 trick_analysis --------------------------------------------------


def analyse_tricks(models: list[ModelData], out_dir: str) -> dict:
    rows: list[dict] = []
    per_author_counter: dict[str, dict[str, int]] = {}
    for m in models:
        per_author_counter[m.alias] = {}
        n_valid = len(m.folds)
        tricks = [f for f in m.folds if f["delta_natural_vs_generated"] < 0]
        n_trick = len(tricks)
        deltas = [f["delta_natural_vs_generated"] for f in m.folds]
        if tricks:
            best = min(tricks, key=lambda f: f["delta_natural_vs_generated"])
            best_author = best["author"]
            best_article = best["target_article_id"]
            best_delta = best["delta_natural_vs_generated"]
        else:
            best_author = ""
            best_article = ""
            best_delta = float("nan")
        rows.append({
            "model": m.alias,
            "n_valid_folds": n_valid,
            "n_trick": n_trick,
            "trick_rate": n_trick / n_valid if n_valid else float("nan"),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "median_delta": float(np.median(deltas)) if deltas else float("nan"),
            "best_trick_author": best_author,
            "best_trick_article": best_article,
            "best_trick_delta": best_delta,
        })
        for f in m.folds:
            author = f["author"]
            per_author_counter[m.alias].setdefault(author, 0)
            if f["delta_natural_vs_generated"] < 0:
                per_author_counter[m.alias][author] += 1

    pooled = [f for m in models for f in m.folds]
    n_pooled = len(pooled)
    n_pooled_trick = sum(1 for f in pooled if f["delta_natural_vs_generated"] < 0)
    rows.append({
        "model": "ALL_MODELS_POOLED",
        "n_valid_folds": n_pooled,
        "n_trick": n_pooled_trick,
        "trick_rate": n_pooled_trick / n_pooled if n_pooled else float("nan"),
        "mean_delta": float(np.mean([f["delta_natural_vs_generated"] for f in pooled]))
            if pooled else float("nan"),
        "median_delta": float(np.median([f["delta_natural_vs_generated"] for f in pooled]))
            if pooled else float("nan"),
        "best_trick_author": "",
        "best_trick_article": "",
        "best_trick_delta": float("nan"),
    })

    csv_path = os.path.join(out_dir, "trick_analysis.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "model,n_valid_folds,n_trick,trick_rate,mean_delta,median_delta,"
            "best_trick_author,best_trick_article,best_trick_delta\n"
        )
        for r in rows:
            f.write(
                f"{r['model']},{r['n_valid_folds']},{r['n_trick']},"
                f"{r['trick_rate']:.4f},{r['mean_delta']:.6f},{r['median_delta']:.6f},"
                f"{r['best_trick_author']},{r['best_trick_article']},"
                f"{r['best_trick_delta']}\n"
            )

    # Trick rate by author -- the easiest-to-impersonate authors.
    author_totals: dict[str, dict[str, int]] = {}
    for m in models:
        for f in m.folds:
            a = f["author"]
            author_totals.setdefault(a, {"tricks": 0, "folds": 0})
            author_totals[a]["folds"] += 1
            if f["delta_natural_vs_generated"] < 0:
                author_totals[a]["tricks"] += 1
    author_rows = sorted(
        (
            {"author": a, "tricks": v["tricks"], "folds": v["folds"],
             "rate": v["tricks"] / v["folds"] if v["folds"] else float("nan")}
            for a, v in author_totals.items()
        ),
        key=lambda r: -r["rate"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    per_model = [r for r in rows if r["model"] != "ALL_MODELS_POOLED"]
    per_model.sort(key=lambda r: -r["trick_rate"])
    ax = axes[0]
    ax.bar([r["model"] for r in per_model], [r["trick_rate"] for r in per_model],
           color="darkorange", edgecolor="black")
    for i, r in enumerate(per_model):
        ax.text(i, r["trick_rate"] + 0.005,
                f"{r['n_trick']}/{r['n_valid_folds']}", ha="center", fontsize=9)
    ax.set_ylabel("Trick Rate  (fraction of folds with delta < 0)")
    ax.set_title("How Often Did Each AI Model Beat the Natural Baseline?")
    ax.set_ylim(0, max(r["trick_rate"] for r in per_model) * 1.25 + 0.02)

    ax = axes[1]
    authors_order = [r["author"] for r in author_rows]
    bottom = np.zeros(len(authors_order))
    cmap = plt.get_cmap("tab10")
    aliases_ord = [m.alias for m in models]
    for i, alias in enumerate(aliases_ord):
        vals = np.array([per_author_counter[alias].get(a, 0) for a in authors_order])
        ax.bar(authors_order, vals, bottom=bottom, color=cmap(i % 10),
               edgecolor="black", label=alias)
        bottom += vals
    ax.set_ylabel("Trick Count (out of 5 folds per author per model)")
    ax.set_title("Per-Author Trick Counts (Stacked by Model)")
    ax.legend(loc="upper right", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(out_dir, "trick_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv": csv_path,
        "plot": path,
        "rows": rows,
        "author_rows": author_rows,
    }


# --- 3 cost_vs_performance --------------------------------------------


def estimate_details_tokens(essays: list[dict], details: list[dict]) -> dict:
    """One-time details stage: per essay, input = essay + assignment-system prompts,
    output = the assignment text."""
    details_by_id = {d["article_id"]: d for d in details if d}
    in_tokens = 0
    out_tokens = 0
    per_essay = []
    for essay in essays:
        eid = essay["article_id"]
        if eid not in details_by_id:
            continue
        msgs = build_details_messages(essay)
        i, _ = messages_to_tokens(msgs)
        assignment = details_by_id[eid].get("assignment", "")
        o = count_tokens(assignment)
        in_tokens += i
        out_tokens += o
        per_essay.append({"article_id": eid, "input_tokens": i, "output_tokens": o})
    return {"input": in_tokens, "output": out_tokens, "per_essay": per_essay}


def estimate_recreate_tokens(
    essays: list[dict], details: list[dict], recreations: list[dict]
) -> dict:
    """Per-model recreate stage: per recreation, input = prompt + corpus, output = body."""
    details_by_id = {d["article_id"]: d for d in details if d}
    essays_by_id = {e["article_id"]: e for e in essays}
    rec_by_id: dict[str, dict] = {}
    for r in recreations:
        src = r.get("source_article_id") or r.get("article_id")
        if src:
            rec_by_id[src] = r
    in_tokens = 0
    out_tokens = 0
    per_call = []
    by_author = {}
    for source_id, rec in rec_by_id.items():
        detail = details_by_id.get(source_id)
        if not detail:
            continue
        target_essay = essays_by_id.get(source_id)
        if not target_essay:
            continue
        target_author = target_essay["author"]
        # All other essays by the same author (mirrors same_author_essays_excluding)
        corpus = [
            e for e in essays
            if e["article_id"] != source_id and e["author"] == target_author
        ]
        msgs = build_recreate_messages(detail, corpus)
        i, _ = messages_to_tokens(msgs)
        body = rec.get("body", "")
        wc = rec.get("word_count") or len(body.split())
        o = count_tokens(body) if body else int(round(wc * 1.33))
        in_tokens += i
        out_tokens += o
        per_call.append({
            "source_article_id": source_id,
            "author": target_author,
            "corpus_size": len(corpus),
            "input_tokens": i,
            "output_tokens": o,
        })
        by_author.setdefault(target_author, {"in": 0, "out": 0, "n": 0})
        by_author[target_author]["in"] += i
        by_author[target_author]["out"] += o
        by_author[target_author]["n"] += 1
    return {
        "input": in_tokens,
        "output": out_tokens,
        "per_call": per_call,
        "by_author": by_author,
    }


def plot_cost_vs_performance(
    cost_rows: list[dict], trick_lookup: dict[str, int], n_valid_lookup: dict[str, int],
    out_dir: str,
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    xs = [r["total_cost_usd"] for r in cost_rows]
    ys = [r["mean_cosine_delta"] for r in cost_rows]
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(cost_rows):
        ax.scatter(xs[i], ys[i], s=110, color=cmap(i % 10),
                   edgecolor="black", zorder=3)
        ax.annotate(r["model"], (xs[i], ys[i]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    if xs:
        ax.axvline(np.median(xs), color="grey", linestyle=":", linewidth=0.8)
        ax.axhline(np.median(ys), color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Mean Cosine Delta  (lower = better mimic)")
    ax.set_title("Cost vs Mimic Quality")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    rates = [trick_lookup[r["model"]] / n_valid_lookup[r["model"]] for r in cost_rows]
    for i, r in enumerate(cost_rows):
        ax.scatter(xs[i], rates[i], s=110, color=cmap(i % 10),
                   edgecolor="black", zorder=3)
        ax.annotate(r["model"], (xs[i], rates[i]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    if xs:
        ax.axvline(np.median(xs), color="grey", linestyle=":", linewidth=0.8)
        ax.axhline(np.median(rates), color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Trick Rate  (fraction of folds with delta < 0)")
    ax.set_title("Cost vs Detection-Evasion Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "cost_vs_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "cost_vs_performance.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "model,details_input_tokens,details_output_tokens,"
            "recreate_input_tokens,recreate_output_tokens,total_tokens,"
            "details_cost_usd,recreate_cost_usd,total_cost_usd,"
            "mean_cosine_delta,trick_rate\n"
        )
        for i, r in enumerate(cost_rows):
            f.write(
                f"{r['model']},{r['details_input_tokens']},{r['details_output_tokens']},"
                f"{r['recreate_input_tokens']},{r['recreate_output_tokens']},"
                f"{r['total_tokens']},"
                f"{r['details_cost_usd']:.6f},{r['recreate_cost_usd']:.6f},"
                f"{r['total_cost_usd']:.6f},"
                f"{r['mean_cosine_delta']:.6f},{rates[i]:.4f}\n"
            )
    return {"plot": path, "csv": csv_path}


def compute_costs(
    models: list[ModelData],
    essays: list[dict],
    details: list[dict],
) -> list[dict]:
    details_tokens = estimate_details_tokens(essays, details)
    rows = []
    for m in models:
        rec_tokens = estimate_recreate_tokens(essays, details, m.recreations)
        # The details cost was paid once with glm-5.2 (the alias used in
        # the assignment stage), but the same assignments are reused by every
        # recreation model. Attribute the upfront cost to glm-5.2 (the actual
        # payer) and report recreate-only cost for the others; include an
        # "amortized" row entry for each model that adds details/4 to its
        # total so the four-model comparison includes the shared overhead.
        inp_price = PRICING[m.alias]["input"]
        out_price = PRICING[m.alias]["output"]
        details_cost = (
            details_tokens["input"] * inp_price / 1_000_000
            + details_tokens["output"] * out_price / 1_000_000
        )
        recreate_cost = (
            rec_tokens["input"] * inp_price / 1_000_000
            + rec_tokens["output"] * out_price / 1_000_000
        )
        # If this model was the one that paid for details, the actual full cost
        # is recreate_cost + details_cost. Otherwise, an "amortized" total is
        # recreate_cost + details_cost / len(models) (best estimate of share).
        amortized_details = details_cost / len(models)
        rows.append({
            "model": m.alias,
            "details_input_tokens": details_tokens["input"],
            "details_output_tokens": details_tokens["output"],
            "recreate_input_tokens": rec_tokens["input"],
            "recreate_output_tokens": rec_tokens["output"],
            "total_tokens": details_tokens["input"] + details_tokens["output"]
                + rec_tokens["input"] + rec_tokens["output"],
            "details_cost_usd": details_cost,
            "recreate_cost_usd": recreate_cost,
            "total_cost_usd": recreate_cost + details_cost if m.alias == "glm-5.2"
                else recreate_cost + amortized_details,
            "mean_cosine_delta": m.summary["overall"]["generated_mean"]
                if m.alias != "glm-5.2" or not np.isnan(m.summary["overall"]["generated_mean"])
                else m.summary["overall"].get("generated_mean", float("nan")),
        })
    # Recompute mean_cosine_delta robustly from kept folds (chatgpt excluded fold)
    for r in rows:
        m = next(x for x in models if x.alias == r["model"])
        gen_deltas = [f["generated_mean_distance"] for f in m.folds
                      if not (isinstance(f["generated_mean_distance"], float)
                              and np.isnan(f["generated_mean_distance"]))]
        if gen_deltas:
            r["mean_cosine_delta"] = float(np.mean(gen_deltas))
        else:
            r["mean_cosine_delta"] = float("nan")
    return rows, details_tokens, [estimate_recreate_tokens(essays, details, m.recreations)
                                  for m in models]


# --- A detection_roc ----------------------------------------------------


def plot_detection_roc(models: list[ModelData], out_dir: str) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab10")
    summary_rows = []

    # Build the unified natural pool from ALL folds of ANY model where
    # natural_mean_distance is finite. We want a single, consistent natural
    # class across models; the natural score is invariant by construction.
    nat_pool: dict[str, float] = {}
    for m in models:
        for f in m.dropped_folds + m.folds:
            tid = f.get("target_article_id")
            v = f.get("natural_mean_distance")
            if tid is not None and v is not None \
                    and not (isinstance(v, float) and np.isnan(v)):
                nat_pool[tid] = v
    nat_arr = np.array(sorted(nat_pool.values()))

    auc_rows = []
    for i, m in enumerate(models):
        gen_scores = []
        for f in m.folds:
            gen = f.get("generated_mean_distance")
            if gen is None or (isinstance(gen, float) and np.isnan(gen)):
                continue
            gen_scores.append(gen)
        if not gen_scores:
            continue
        gen_arr = np.array(gen_scores)

        # AUC via Mann-Whitney U -- treat higher Cosine Delta score as "AI detected".
        n_pos = gen_arr.size
        n_neg = nat_arr.size
        all_scores = np.concatenate([nat_arr, gen_arr])
        order = np.argsort(all_scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, all_scores.size + 1)
        sum_ranks_pos = ranks[n_neg:].sum()  # positives are appended after negatives
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = float(auc)

        # Build ROC curve by threshold sweep.
        thresholds = np.unique(np.concatenate([nat_arr, gen_arr]))
        margins = np.concatenate([thresholds - 1e-9, thresholds + 1e-9])
        thresholds = np.unique(np.concatenate([thresholds, margins]))
        tprs = []
        fprs = []
        for t in thresholds:
            tp = np.sum(gen_arr >= t)
            fn = np.sum(gen_arr < t)
            fp = np.sum(nat_arr >= t)
            tn = np.sum(nat_arr < t)
            tprs.append(tp / (tp + fn) if (tp + fn) else 0)
            fprs.append(fp / (fp + tn) if (fp + tn) else 0)
        ax.plot(fprs, tprs, color=cmap(i % 10), lw=2,
                label=f"{m.alias}  (AUC = {auc:.3f}, n_pos={n_pos})")
        auc_rows.append({"model": m.alias, "auc": auc,
                         "n_pos": int(n_pos), "n_neg": int(n_neg)})

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate (natural classified as AI)")
    ax.set_ylabel("True Positive Rate (AI detected)")
    ax.set_title(f"Detection ROC: Cosine Delta as Natural-vs-AI Score  ({len(nat_arr)} natural vs per-model AI folds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "detection_roc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "detection_roc.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("model,auc,n_pos,n_neg\n")
        for r in auc_rows:
            f.write(f"{r['model']},{r['auc']:.6f},{r['n_pos']},{r['n_neg']}\n")
    return {"plot": path, "csv": csv_path, "rows": auc_rows}


# --- B per_author_heatmap ----------------------------------------------


def plot_per_author_heatmap(summary_multi: dict, out_dir: str) -> dict:
    aliases = sorted(summary_multi["per_model"].keys())
    authors: set[str] = set()
    for alias in aliases:
        authors.update(summary_multi["per_model"][alias].get("per_author", {}).keys())
    authors = sorted(authors)

    # Sort authors by natural_mean desc (easiest = least detectable)
    sample = summary_multi["per_model"][aliases[0]]["per_author"]
    nat_means = {a: sample[a]["natural_mean"] for a in authors}
    authors = sorted(authors, key=lambda a: -nat_means[a])

    # Sort models by overall delta asc (best mimic first)
    overall_delta = {a: summary_multi["per_model"][a]["overall"].get("delta_mean") or float("inf")
                     for a in aliases}
    aliases = sorted(aliases, key=lambda a: overall_delta[a])

    M = np.zeros((len(authors), len(aliases)))
    for j, alias in enumerate(aliases):
        per_author = summary_multi["per_model"][alias]["per_author"]
        for i, author in enumerate(authors):
            v = per_author.get(author, {}).get("delta_mean")
            M[i, j] = float("nan") if v is None else float(v)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(len(aliases)))
    ax.set_xticklabels(aliases)
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels(authors)
    ax.set_title("Per-Author Delta_Mean (AI - Natural) Across Models")
    fig.colorbar(im, ax=ax, label="delta_mean (red = easily detected, blue = beat natural)")
    fig.tight_layout()
    path = os.path.join(out_dir, "per_author_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "per_author_heatmap.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("author," + ",".join(aliases) + "\n")
        for i, a in enumerate(authors):
            row = [a]
            for j in range(len(aliases)):
                v = M[i, j]
                row.append(f"{v:.6f}" if not np.isnan(v) else "")
            f.write(",".join(row) + "\n")
    return {"plot": path, "csv": csv_path}


# --- C model_agreement --------------------------------------------------


def _build_trick_matrix(models: list[ModelData]) -> tuple[list[str], list[str], np.ndarray]:
    """Build the (models x targets) trick matrix shared by both plots."""
    aliases = [m.alias for m in models]
    target_order = sorted({f["target_article_id"]
                           for m in models for f in (m.folds + m.dropped_folds)})
    targets: list[str] = []
    seen = set()
    for m in models:
        for f in m.folds + m.dropped_folds:
            tid = f["target_article_id"]
            if tid in target_order and tid not in seen:
                targets.append(tid)
                seen.add(tid)

    M = np.zeros((len(aliases), len(targets)), dtype=int)
    for i, m in enumerate(models):
        for j, tid in enumerate(targets):
            for f in (m.folds + m.dropped_folds):
                if f["target_article_id"] == tid:
                    delta = f.get("delta_natural_vs_generated")
                    if delta is not None \
                            and not (isinstance(delta, float) and np.isnan(delta)) \
                            and delta < 0:
                        M[i, j] = 1
                    break
    return aliases, targets, M


def plot_model_agreement(models: list[ModelData], out_dir: str) -> dict:
    aliases, targets, M = _build_trick_matrix(models)

    fig, ax = plt.subplots(figsize=(11, 4.0))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels(aliases)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.split("/")[-1] for t in targets], rotation=80, fontsize=7)
    ax.set_title("Per-Essay Trick Map  (1 = AI beat the natural baseline on that fold)")
    fig.colorbar(im, ax=ax, label="Trick?")
    fig.tight_layout()
    path = os.path.join(out_dir, "model_agreement.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"plot": path, "matrix": M, "aliases": aliases, "targets": targets}


def plot_jaccard_overlap(models: list[ModelData], out_dir: str) -> dict:
    aliases, _, M = _build_trick_matrix(models)

    jaccard = np.zeros((len(aliases), len(aliases)))
    for i in range(len(aliases)):
        for j in range(len(aliases)):
            a = set(np.where(M[i] == 1)[0])
            b = set(np.where(M[j] == 1)[0])
            union = a | b
            jaccard[i, j] = len(a & b) / len(union) if union else 0.0

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(jaccard, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(aliases)))
    ax.set_xticklabels(aliases, rotation=20, ha="right")
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels(aliases)
    ax.set_title("Jaccard Overlap of Trick Sets Between Models")
    for i in range(len(aliases)):
        for j in range(len(aliases)):
            ax.text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center",
                    color="white" if jaccard[i, j] > 0.5 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="Jaccard")
    fig.tight_layout()
    path = os.path.join(out_dir, "jaccard_overlap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_dir, "model_agreement_jaccard.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("," + ",".join(aliases) + "\n")
        for i, alias in enumerate(aliases):
            row = [alias] + [f"{jaccard[i, j]:.4f}" for j in range(len(aliases))]
            f.write(",".join(row) + "\n")
    return {"plot": path, "csv": csv_path, "jaccard": jaccard}


# --- D delta_distribution -----------------------------------------------


def plot_delta_distribution(models: list[ModelData], out_dir: str) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for m in models:
        vals = [f["delta_natural_vs_generated"] for f in m.folds
                if f.get("delta_natural_vs_generated") is not None
                and not (isinstance(f["delta_natural_vs_generated"], float)
                         and np.isnan(f["delta_natural_vs_generated"]))]
        if vals:
            data.append(vals)
            labels.append(f"{m.alias}\n(n={len(vals)})")
    cmap = plt.get_cmap("tab10")
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.7)
    for pc, color in zip(parts["bodies"], [cmap(i % 10) for i in range(len(data))]):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, label="natural baseline (delta=0)")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine Delta (AI - Natural)")
    ax.set_title("Per-Model Distribution of Detection Margins")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "delta_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"plot": path}


# --- final_report.md ----------------------------------------------------


def write_final_report(
    out_dir: str,
    ranking_rows: list[dict],
    trick_rows: list[dict],
    trick_author_rows: list[dict],
    auc_rows: list[dict],
    cost_rows: list[dict],
    summary_multi: dict,
    dropped_report: list[tuple[str, list[dict]]],
) -> str:
    per_model = [r for r in trick_rows if r["model"] != "ALL_MODELS_POOLED"]
    cheapest = min(cost_rows, key=lambda r: r["total_cost_usd"])
    best_mimic = min(ranking_rows, key=lambda r: r["mean"] if r["kind"] != "natural" else float("inf"))
    easiest_author = max(trick_author_rows, key=lambda r: r["rate"])
    hardest_author = min(trick_author_rows, key=lambda r: r["rate"])
    best_roc = max(auc_rows, key=lambda r: r["auc"])

    pricing_table = (
        "| Model | Input $/1M | Output $/1M |\n"
        "|---|---|---|\n"
    )
    for alias, p in PRICING.items():
        pricing_table += f"| `{alias}` | {p['input']:.3f} | {p['output']:.3f} |\n"

    cost_table = (
        "| Model | Re-create input | Re-create output | Details overhead (paid once via glm) | "
        "Recreate $ | Total $ |\n"
        "|---|---|---|---|---|\n"
    )
    for r in cost_rows:
        cost_table += (
            f"| `{r['model']}` | {r['recreate_input_tokens']:,} | "
            f"{r['recreate_output_tokens']:,} | "
            f"~{r['details_cost_usd'] / len(cost_rows):.2f} | "
            f"{r['recreate_cost_usd']:.2f} | "
            f"**{r['total_cost_usd']:.2f}** |\n"
        )

    trick_table = (
        "| Model | n_folds | trick count | trick rate | mean delta | median delta |\n"
        "|---|---|---|---|---|---|\n"
    )
    for r in per_model:
        trick_table += (
            f"| `{r['model']}` | {r['n_valid_folds']} | {r['n_trick']} | "
            f"{r['trick_rate']:.3f} | {r['mean_delta']:.3f} | {r['median_delta']:.3f} |\n"
        )
    pooled = next(r for r in trick_rows if r["model"] == "ALL_MODELS_POOLED")
    trick_table += (
        f"| **ALL_MODELS_POOLED** | {pooled['n_valid_folds']} | {pooled['n_trick']} | "
        f"{pooled['trick_rate']:.3f} | {pooled['mean_delta']:.3f} | "
        f"{pooled['median_delta']:.3f} |\n"
    )

    auc_table = (
        "| Model | AUC | AI folds (positive class) | Natural folds (negative class) |\n"
        "|---|---|---|---|\n"
    )
    for r in auc_rows:
        auc_table += (
            f"| `{r['model']}` | {r['auc']:.3f} | {r['n_pos']} | {r['n_neg']} |\n"
        )

    dropped_md = ""
    if dropped_report:
        dropped_md = "\n## Caveats\n\n"
        for alias, dropped in dropped_report:
            if not dropped:
                continue
            dropped_md += f"- `{alias}`: dropped {len(dropped)} fold(s) from aggregates because the model returned an essay too short to chunk (minimum fill ratio on 1000-token chunks):\n"
            for d in dropped:
                dropped_md += (
                    f"    - fold {d['fold']} - {d['author']} - "
                    f"`{d['target_article_id']}`\n"
                )

    md = f"""# Final Analysis - Stylometry vs AI Generation

This report consolidates the four-model OpenRouter cross-validation
(`chatgpt-5.6`, `claude-opus-4.8`, `deepseek-v4-pro`, `glm-5.2`).

## Headline numbers

- **Cheapest model (amortized total):** `{cheapest['model']}` at
  **${cheapest['total_cost_usd']:.2f}** for the 25-fold recreation run.
- **Closest mimic of natural style (lowest mean Cosine Delta):**
  `{best_mimic['entity']}` at {best_mimic['mean']:.3f}.
- **Easiest target author to impersonate (most tricks across all models):**
  `{easiest_author['author']}` with a trick rate of
  {easiest_author['rate']:.1%} ({easiest_author['tricks']}/{easiest_author['folds']}).
- **Hardest target author to impersonate (fewest tricks):**
  `{hardest_author['author']}` with a trick rate of
  {hardest_author['rate']:.1%} ({hardest_author['tricks']}/{hardest_author['folds']}).
- **Detection AUC champion:** `{best_roc['model']}` at {best_roc['auc']:.3f}
  (higher = stylometry detected the AI essay more reliably from its
  Cosine Delta score alone).

## Cost assumptions

{PRICING_NOTE}

{pricing_table}

Token counts are reconstructed from the exact prompts shipped to the
OpenRouter SDK (mirroring `src/generate.py`), since the recreation JSON
did not persist the provider usage payload. Token counts use
`cl100k_base` tiktoken when available; otherwise an NLTK
word-tokenizer with a 1.33 tokens/word fallback.

The details-stage (writing assignments) is paid once and reused by all
four re-creation models. Its cost is attributed to `glm-5.2` (the model
that actually generated the assignments) and amortized
(divide by 4) for the comparison table below.

{cost_table}

## Trick analysis (delta < 0)

AI "tricks" the stylometry model when the recreated essay's mean
Cosine Delta to the four-essay same-author corpus is *lower* than the
natural target's own distance - i.e., the detector would classify the
AI essay as the target author before the real one. Numbers are based
on the kept folds (chatgpt-5.6 fold 3 dropped, see Caveats).

{trick_table}

Authors ranked by trick rate (across all four models):

| Author | Tricks | Folds | Rate |
|---|---|---|---|
""" + "\n".join(
        f"| `{r['author']}` | {r['tricks']} | {r['folds']} | {r['rate']:.1%} |"
        for r in trick_author_rows
    ) + f"""

## Detection ROC

Stylometry scores each fold by Cosine Delta; we treat "AI >= threshold"
as the prediction and compute ROC against the 25 natural folds as
negatives.

{auc_table}

## Plots

- ![Generalized Ranking](ranking_cosine_delta.png)
- ![Author vs AI Closeness](ranking_cosine_match.png)
- ![Per-Author Breakdown](author_breakdown.png)
- ![Trick Analysis](trick_analysis.png)
- ![Cost vs Performance](cost_vs_performance.png)
- ![Detection ROC](detection_roc.png)
- ![Per-Author Heatmap](per_author_heatmap.png)
- ![Model Agreement](model_agreement.png)
- ![Jaccard Overlap](jaccard_overlap.png)
- ![Delta Distribution](delta_distribution.png)
{dropped_md}
## Methodology notes

- **Effect-size distance, not p-values:** the metric is Cosine Delta on
  the 500 most-frequent-word z-scored reference, per the project
  pipeline. Lower = closer in style.
- **Leakage-safe scoring:** per fold, MFW vocabulary and reference
  z-scores are fit on natural chunks only; AI chunks never participate
  in fitting.
- **Same-author corpus of 4 essays:** every fold leaves one
  target essay out and feeds the recreation prompt with the other
  four per-author essays. (In the actual generation call, the corpus
  is the full set of other essays by that author in `essays.json` -
  hence the larger recreation input token counts.)
- **Chunking policy:** 1000-token chunks at 0.8 fill-ratio. Any AI
  recreation shorter than 800 tokens after word tokenisation yields
  zero chunks for that fold and is dropped.
"""
    path = os.path.join(out_dir, "final_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


# --- main ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final cross-model stylometry analysis.")
    p.add_argument("--out-root", default=os.path.join(REPO_ROOT, "src", "plots"))
    p.add_argument("--generated-dir", default=os.path.join(REPO_ROOT, "generated"))
    p.add_argument("--out-dir-name", default="final")
    p.add_argument("--essays", default=os.path.join(REPO_ROOT, "essays.json"))
    p.add_argument("--details", default=os.path.join(REPO_ROOT, "generated", "essay_details.json"))
    p.add_argument("--models-json", default=os.path.join(REPO_ROOT, "models.json"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    essays = load_json(args.essays, default=[])
    details = load_json(args.details, default=[])
    models_all = load_json(args.models_json, default={})

    out_dir = os.path.join(args.out_root, "experiment_multi", args.out_dir_name)
    ensure_dir(out_dir)

    aliases = sorted(models_all)
    models = [load_model_data(a, args.out_root, args.generated_dir) for a in aliases]

    summary_multi_path = os.path.join(args.out_root, "experiment_multi", "summary.json")
    summary_multi = load_json(summary_multi_path, default={})

    dropped_report = [(m.alias, m.dropped_folds) for m in models]

    # 1a
    natural_overall = pooled_natural(models)
    ranking = plot_ranking(models, natural_overall, out_dir)
    ranking_match = plot_ranking_match(models, natural_overall, out_dir)

    # 1b
    author_breakdown = plot_author_breakdown(models, summary_multi, out_dir)

    # 2
    trick = analyse_tricks(models, out_dir)
    trick_lookup = {r["model"]: r["n_trick"] for r in trick["rows"] if r["model"] != "ALL_MODELS_POOLED"}
    n_valid_lookup = {r["model"]: r["n_valid_folds"] for r in trick["rows"] if r["model"] != "ALL_MODELS_POOLED"}

    # 3
    cost_rows, details_tokens, per_model_rec_tokens = compute_costs(models, essays, details)
    cost_plot = plot_cost_vs_performance(cost_rows, trick_lookup, n_valid_lookup, out_dir)

    # A
    roc = plot_detection_roc(models, out_dir)

    # B
    heatmap = plot_per_author_heatmap(summary_multi, out_dir)

    # C
    agreement = plot_model_agreement(models, out_dir)
    jaccard = plot_jaccard_overlap(models, out_dir)

    # D
    delta_dist = plot_delta_distribution(models, out_dir)

    # final_report.md
    report_md = write_final_report(
        out_dir=out_dir,
        ranking_rows=ranking["rows"],
        trick_rows=trick["rows"],
        trick_author_rows=trick["author_rows"],
        auc_rows=roc["rows"],
        cost_rows=cost_rows,
        summary_multi=summary_multi,
        dropped_report=dropped_report,
    )

    # headline summary on stdout
    print("=" * 78)
    print("Final analysis written to:", out_dir)
    print("=" * 78)
    print()
    print(f"{'model':<18} {'kept':>5} {'tricks':>7} {'rate':>7} "
          f"{'mean_delta':>11} {'cost_usd':>10} {'AUC':>7}")
    for m in models:
        r = next(r for r in trick["rows"] if r["model"] == m.alias)
        c = next(r for r in cost_rows if r["model"] == m.alias)
        a = next(r for r in roc["rows"] if r["model"] == m.alias)
        print(f"{m.alias:<18} {r['n_valid_folds']:>5} {r['n_trick']:>7} "
              f"{r['trick_rate']:>7.3f} {r['mean_delta']:>11.3f} "
              f"{c['total_cost_usd']:>10.2f} {a['auc']:>7.3f}")
    print()
    print(f"Cheapest model (amortized): {min(cost_rows, key=lambda r: r['total_cost_usd'])['model']}")
    print(f"Closest mimic (lowest mean Cosine Delta): "
          f"{min(ranking['rows'], key=lambda r: r['mean'] if r['kind'] != 'natural' else float('inf'))['entity']}")
    print()
    print("Report:", report_md)
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

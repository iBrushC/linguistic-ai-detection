"""Generate AI-recreated essays via OpenRouter.

Workflow:

1. ``python -m src.generate details --model glm-5.2``
   Produces one shared writing assignment per source essay at
   ``generated/essay_details.json``. Each assignment is derived from the
   original essay's title and text.

2. ``python -m src.generate recreate --essay-index ...``
   Uses ``glm-5.2`` to recreate an essay from an assignment plus the
   same-author natural reference essays. The recreation sees the four
   corpus essays only; the held-out natural essay is never supplied.
   Results are written to ``generated/essays_<model>_<count>.json``.

Each generated record contains the source ``article_id``, the natural
target author, the recreation model, the prompt fingerprints, the fold
context, and the produced text. The same schema works with the chunking
and experiment modules in this package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import experiment as experiment_mod  # noqa: E402  (path tweak above)


ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
MODELS_PATH = os.path.join(REPO_ROOT, "models.json")
GENERATED_DIR = os.path.join(REPO_ROOT, "generated")
DETAILS_PATH = os.path.join(GENERATED_DIR, "essay_details.json")

DEFAULT_MAX_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 1.0

DEFAULT_MODEL = "glm-5.2"


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


# --- IO helpers ----------------------------------------------------------


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_models() -> dict[str, str]:
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(f"models.json not found at {MODELS_PATH}")
    with open(MODELS_PATH, encoding="utf-8") as f:
        models = json.load(f)
    if not isinstance(models, dict) or not models:
        raise ValueError("models.json must be a non-empty object")
    return models


def load_essays() -> list[dict]:
    data = load_json(ESSAYS_PATH, default=[])
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{ESSAYS_PATH} is missing or empty")
    return data


def author_name(essay: dict) -> str:
    return str(essay["author"]).strip()


def article_title(essay: dict, fallback: str) -> str:
    title = essay.get("title") or essay.get("article_title") or fallback
    return str(title).strip()


def word_count(text: str) -> int:
    return len(text.split())


# --- OpenRouter client ---------------------------------------------------


def make_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY env var is not set. Set it before running generate.py."
        )
    from openrouter import OpenRouter

    kwargs: dict = {"api_key": api_key}
    referer = os.environ.get("OPENROUTER_REFERER")
    title = os.environ.get("OPENROUTER_TITLE")
    if referer:
        kwargs["http_referer"] = referer
    if title:
        kwargs["app_title"] = title
    return OpenRouter(**kwargs)


def chat(
    client,
    messages: Sequence[dict],
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.send(
                model=model,
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("model returned empty content")
            return content.strip()
        except TypeError as e:
            raise SystemExit(
                f"OpenRouter SDK call signature rejected ({e!r}); "
                "check that openrouter>=0.3.0 is installed."
            )
        except Exception as e:  # pragma: no cover - network behaviour
            last_exc = e
            sleep_s = RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                f"  [retry] {model} attempt {attempt + 1}/{MAX_RETRIES} failed: "
                f"{type(e).__name__}: {e}; sleeping {sleep_s}s",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"chat failed after {MAX_RETRIES} retries: {last_exc!r}")


# --- Public generation functions -----------------------------------------


def build_assignment_prompt(essay: dict) -> tuple[str, int]:
    """Build the user-turn prompt for assignment generation."""
    title = article_title(essay, fallback=essay["article_id"])
    wc = word_count(essay["text"])
    user = (
        f"Title: {title}\n"
        f"Author: {author_name(essay)}\n\n"
        f"Target length for the recreated piece: approximately {wc} words "
        f"(a goal, not strict).\n\n"
        f"Essay:\n{essay['text']}"
    )
    return user, wc


def generate_essay_details(essay: dict, client, model_slug: str) -> dict:
    user, wc = build_assignment_prompt(essay)
    messages = [
        {"role": "system", "content": SYSTEM_DETAILS},
        {"role": "system", "content": DISCLAIMER},
        {"role": "user", "content": user},
    ]
    assignment = chat(client, messages, model_slug)
    return {
        "article_id": essay["article_id"],
        "author": essay["author"],
        "title": article_title(essay, fallback=essay["article_id"]),
        "word_count": wc,
        "assignment": assignment,
        "model_slug": model_slug,
    }


def build_recreate_prompt(
    detail: dict,
    same_author_essays: Sequence[dict],
) -> str:
    author = author_name(detail)
    if same_author_essays:
        corpus = "\n\n".join(
            "--- " + article_title(e, fallback=e["article_id"]) + " ---\n" + e["text"]
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
    return user


def generate_recreation(
    detail: dict,
    same_author_essays: Sequence[dict],
    client,
    model_slug: str,
    source_article_id: str,
) -> dict:
    user = build_recreate_prompt(detail, same_author_essays)
    messages = [
        {"role": "system", "content": SYSTEM_RECREATE},
        {"role": "user", "content": user},
    ]
    body = chat(client, messages, model_slug)
    return {
        "article_id": source_article_id,
        "source_article_id": source_article_id,
        "author": detail["author"],
        "title": detail.get("title", source_article_id),
        "body": body,
        "model_slug": model_slug,
        "model_alias": detail.get("model_alias", ""),
        "word_count": word_count(body),
    }


def same_author_essays_excluding(essays: list[dict], target_article_id: str) -> list[dict]:
    target = next((e for e in essays if e["article_id"] == target_article_id), None)
    if target is None:
        raise SystemExit(f"unknown article_id {target_article_id!r}")
    target_author = author_name(target)
    return [
        e for e in essays
        if e["article_id"] != target_article_id and author_name(e) == target_author
    ]


# --- Subcommand handlers -------------------------------------------------


def _select_model(models: dict[str, str], name: str | None) -> tuple[str, str]:
    if name is None:
        if DEFAULT_MODEL not in models:
            raise SystemExit(
                f"default model {DEFAULT_MODEL!r} not in models.json; "
                "pass --model explicitly."
            )
        name = DEFAULT_MODEL
    if name not in models:
        raise SystemExit(f"unknown model {name!r}; choices: {sorted(models)}")
    return name, models[name]


def cmd_details(args) -> None:
    essays = load_essays()
    models = load_models()
    model_name, model_slug = _select_model(models, args.model)

    client = make_client()

    existing = load_json(DETAILS_PATH, default=[])
    if not isinstance(existing, list):
        existing = []
    by_id = {item.get("article_id"): item for item in existing if item}

    indices = (
        range(len(essays)) if args.essay_index is None else [args.essay_index]
    )

    for idx in indices:
        if idx < 0 or idx >= len(essays):
            print(f"[details] index {idx} out of range", file=sys.stderr)
            continue
        essay = essays[idx]
        article_id = essay["article_id"]
        if article_id in by_id and by_id[article_id] is not None:
            print(f"[details] article_id={article_id}: already populated, skipping")
            continue
        preview = article_title(essay, fallback=article_id)[:60]
        print(f"[details] {article_id}: {preview}... via {model_name}")
        detail = generate_essay_details(essay, client, model_slug)
        detail["model_alias"] = model_name
        by_id[article_id] = detail

    ordered = [by_id.get(e["article_id"]) for e in essays]
    save_json_atomic(DETAILS_PATH, ordered)
    print(f"[details] saved to {DETAILS_PATH}")


def _build_recreation_inputs(args) -> tuple[list[dict], list[dict], dict[str, str], str, str]:
    essays = load_essays()
    details_list = load_json(DETAILS_PATH, default=[])
    if not isinstance(details_list, list) or not details_list:
        raise SystemExit(
            f"{DETAILS_PATH} is missing or empty; run `python -m src.generate details` first."
        )
    details_by_id = {d.get("article_id"): d for d in details_list if d}
    missing = [d for d in details_by_id.values() if not d]
    if missing:
        raise SystemExit(f"{DETAILS_PATH} has empty entries; rerun the details stage.")

    models = load_models()
    model_name, model_slug = _select_model(models, args.model)
    return essays, list(details_by_id.values()), models, model_name, model_slug


def cmd_recreate(args) -> None:
    essays, details_by_id_values, models, model_name, model_slug = _build_recreation_inputs(args)
    details_by_id = {d["article_id"]: d for d in details_by_id_values}
    client = make_client()

    if args.use_selector:
        manifest = experiment_mod.load_manifest(args.use_selector)
        target_ids = [e["article_id"] for e in manifest["essays"]]
    else:
        target_ids = [e["article_id"] for e in essays]

    if args.essay_index is not None:
        target_ids = [target_ids[args.essay_index]]

    out_path = os.path.join(
        GENERATED_DIR, f"essays_{model_name}_recreate.json"
    )
    existing = load_json(out_path, default=[])
    if not isinstance(existing, list):
        existing = []
    by_source: dict[str, dict] = {item.get("source_article_id"): item for item in existing if item}

    def produce(article_id: str) -> dict:
        detail = details_by_id.get(article_id)
        if detail is None:
            raise SystemExit(f"no assignment for {article_id}; run details first")
        corpus_essays = same_author_essays_excluding(essays, article_id)
        return generate_recreation(
            detail=detail,
            same_author_essays=corpus_essays,
            client=client,
            model_slug=model_slug,
            source_article_id=article_id,
        )

    def commit(article_id: str, result: dict | None) -> None:
        if result is None:
            return
        by_source[article_id] = result
        ordered = [by_source.get(aid) for aid in target_ids]
        save_json_atomic(out_path, ordered)
        print(f"[recreate] {model_name}: {article_id}: wrote essay")

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(produce, aid): aid for aid in target_ids}
            for fut in as_completed(futures):
                aid = futures[fut]
                commit(aid, fut.result())
    else:
        for aid in target_ids:
            if aid in by_source and by_source[aid] is not None and not args.overwrite:
                print(f"[recreate] {model_name}: {aid}: already present, skipping")
                continue
            commit(aid, produce(aid))

    print(f"[recreate] {model_name}: saved to {out_path}")


def cmd_list_models(args) -> None:
    models = load_models()
    for name, slug in models.items():
        print(f"{name}\t{slug}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AI-recreated essays mimicking essay authors via OpenRouter.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models defined in models.json and exit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_details = sub.add_parser(
        "details", help="Generate essay_details.json (one assignment per essay)."
    )
    p_details.add_argument(
        "--model",
        default=None,
        help=f"Friendly model name from models.json (default: {DEFAULT_MODEL}).",
    )
    p_details.add_argument(
        "--essay-index", type=int, default=None, help="Only generate for this essay index."
    )

    p_recreate = sub.add_parser(
        "recreate",
        help="Recreate essays with author context using the chosen model.",
    )
    p_recreate.add_argument("--model", default=None, help=f"Friendly model name (default: {DEFAULT_MODEL}).")
    p_recreate.add_argument(
        "--use-selector",
        default=None,
        help="Path to an experiment manifest restricting targets to the selected essays.",
    )
    p_recreate.add_argument(
        "--essay-index", type=int, default=None, help="Only recreate this essay index."
    )
    p_recreate.add_argument(
        "--workers", type=int, default=1, help="Concurrent workers (default 1)."
    )
    p_recreate.add_argument(
        "--overwrite", action="store_true", help="Recreate even if a result already exists."
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        cmd_list_models(args)
        return 0

    if args.command == "details":
        cmd_details(args)
    elif args.command == "recreate":
        cmd_recreate(args)
    else:  # pragma: no cover - argparse enforces required subcommand
        parser.error("missing subcommand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
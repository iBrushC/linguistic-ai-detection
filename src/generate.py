# Generates synthetic essays that mimic the authors in essays.json, via OpenRouter.
#
# CLI:
#   python generate.py --list-models
#   python generate.py details [--model NAME] [--essay-index N]
#   python generate.py naive   --model NAME | --all [--essay-index N] [--workers W]
#   python generate.py full    --model NAME | --all [--essay-index N] [--workers W]
#
# Outputs:
#   generated/essay_details.json
#   generated/essays_<model>_<count>.json

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import get_all_metrics


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")
MODELS_PATH = os.path.join(REPO_ROOT, "models.json")
GENERATED_DIR = os.path.join(REPO_ROOT, "generated")
DETAILS_PATH = os.path.join(GENERATED_DIR, "essay_details.json")

DEFAULT_MAX_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 1.0
DETAILS_DEFAULT_MODEL = "chatgpt-5.6"


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
    "Output only the assignment — no preamble, no labels, no commentary."
)

SYSTEM_NAIVE = DISCLAIMER

SYSTEM_FULL = (
    DISCLAIMER
    + "\n\n"
    "Your task is to imitate the author's style as closely as possible. Reference "
    "essays by the same author are provided below, along with a list of stylometric "
    "statistics under test. Write so the result falls within the author's observed "
    "distributions."
)


# --- IO helpers -----------------------------------------------------------

def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, obj):
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


def load_models():
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(f"models.json not found at {MODELS_PATH}")
    with open(MODELS_PATH, encoding="utf-8") as f:
        models = json.load(f)
    if not isinstance(models, dict) or not models:
        raise ValueError("models.json must be a non-empty object")
    return models


def load_essays():
    data = load_json(ESSAYS_PATH, default=[])
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{ESSAYS_PATH} is missing or empty")
    return data


def author_name(essay):
    return essay["author"].replace("By ", "").strip()


def word_count(text):
    return len(text.split())


_METRIC_NAMES = None


def get_metric_names():
    global _METRIC_NAMES
    if _METRIC_NAMES is None:
        sample = (
            "This is a sample paragraph. It has several sentences, with a few clauses "
            "of varying length. The model should write essays with similar distribution."
        )
        _METRIC_NAMES = sorted(get_all_metrics(sample).keys())
    return _METRIC_NAMES


# --- OpenRouter client ----------------------------------------------------

def make_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY env var is not set. Set it before running generate.py."
        )
    from openrouter import OpenRouter

    kwargs = {"api_key": api_key}
    referer = os.environ.get("OPENROUTER_REFERER")
    title = os.environ.get("OPENROUTER_TITLE")
    if referer:
        kwargs["http_referer"] = referer
    if title:
        kwargs["app_title"] = title
    return OpenRouter(**kwargs)


def chat(client, messages, model, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.send(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("model returned empty content")
            return content.strip()
        except TypeError as e:
            raise SystemExit(f"OpenRouter SDK call signature rejected ({e!r}); "
                             f"check that openrouter>=0.3.0 is installed.")
        except Exception as e:
            last_exc = e
            sleep_s = RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                f"  [retry] {model} attempt {attempt + 1}/{MAX_RETRIES} failed: "
                f"{type(e).__name__}: {e}; sleeping {sleep_s}s",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"chat failed after {MAX_RETRIES} retries: {last_exc!r}")


# --- Public generation functions ------------------------------------------

def generate_essay_details(essay, client, model_slug):
    wc = word_count(essay["body"])
    user = (
        f"Title: {essay['title'].strip()}\n"
        f"Author: {author_name(essay)}\n\n"
        f"Target length for the recreated piece: approximately {wc} words "
        f"(a goal, not strict).\n\n"
        f"Essay:\n{essay['body']}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_DETAILS},
        {"role": "user", "content": user},
    ]
    assignment = chat(client, messages, model_slug)
    return {
        "title": essay["title"].strip(),
        "author": essay["author"],
        "word_count": wc,
        "assignment": assignment,
    }


def generate_essay_naive(detail, client, model_slug):
    user = (
        f"Write the essay described by the following assignment. Target length: "
        f"approximately {detail['word_count']} words (a goal, not strict). "
        f"Output only the essay itself — no preamble, no commentary, no labels.\n\n"
        f"Assignment:\n{detail['assignment']}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_NAIVE},
        {"role": "user", "content": user},
    ]
    body = chat(client, messages, model_slug)
    return {
        "title": detail["title"],
        "author": detail["author"],
        "body": body,
    }


def generate_essay_full(detail, same_author_essays, metric_names, client, model_slug):
    author = author_name(detail)
    if same_author_essays:
        corpus = "\n\n".join(
            f"--- {e['title'].strip()} ---\n{e['body']}"
            for e in same_author_essays
        )
    else:
        corpus = "(No other essays by this author are available.)"
    metrics_csv = ", ".join(metric_names)

    user = (
        f"Write the essay described by the following assignment. Target length: "
        f"approximately {detail['word_count']} words (a goal, not strict). "
        f"Match the author's style as closely as possible — pay particular attention "
        f"to the listed statistics and write so they fall within the distributions you "
        f"observe. Output only the essay itself — no preamble, no commentary, no labels.\n\n"
        f"Assignment:\n{detail['assignment']}\n\n"
        f"Reference essays by {author}:\n\n{corpus}\n\n"
        f"Stylometric statistics under test:\n{metrics_csv}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_FULL},
        {"role": "user", "content": user},
    ]
    body = chat(client, messages, model_slug)
    return {
        "title": detail["title"],
        "author": detail["author"],
        "body": body,
    }


# --- Resume / count helpers ----------------------------------------------

def same_author_essays_excluding(essays, target_index):
    target_author = author_name(essays[target_index])
    return [
        e for i, e in enumerate(essays)
        if i != target_index and author_name(e) == target_author
    ]


def essays_path(model_name, count):
    return os.path.join(GENERATED_DIR, f"essays_{model_name}_{count}.json")


def find_target_count(model_name, n_essays):
    """Pick the smallest existing count whose file is incomplete; otherwise next free."""
    prefix = f"essays_{model_name}_"
    used = []
    if os.path.isdir(GENERATED_DIR):
        for name in os.listdir(GENERATED_DIR):
            if name.startswith(prefix) and name.endswith(".json"):
                mid = name[len(prefix):-len(".json")]
                try:
                    used.append(int(mid))
                except ValueError:
                    continue
    for c in sorted(used):
        data = load_json(essays_path(model_name, c), default=[])
        if (not isinstance(data, list)
                or len(data) < n_essays
                or any(e is None for e in data)):
            return c
    return (max(used) + 1) if used else 0


# --- Subcommand handlers --------------------------------------------------

def cmd_details(args):
    models = load_models()
    if args.model is None:
        if DETAILS_DEFAULT_MODEL not in models:
            raise SystemExit(
                f"default model {DETAILS_DEFAULT_MODEL!r} not in models.json; "
                f"pass --model explicitly."
            )
        model_name = DETAILS_DEFAULT_MODEL
    else:
        if args.model not in models:
            raise SystemExit(f"unknown model {args.model!r}; choices: {sorted(models)}")
        model_name = args.model
    model_slug = models[model_name]

    client = make_client()
    essays = load_essays()

    existing = load_json(DETAILS_PATH, default=[])
    if not isinstance(existing, list):
        existing = []

    indices = range(len(essays)) if args.essay_index is None else [args.essay_index]

    for idx in indices:
        if idx < 0 or idx >= len(essays):
            print(f"[details] index {idx} out of range", file=sys.stderr)
            continue
        if idx < len(existing) and existing[idx] is not None:
            print(f"[details] idx={idx}: already populated, skipping")
            continue
        title_preview = essays[idx]["title"].strip().replace("\n", " ")[:60]
        print(f"[details] idx={idx}: {title_preview}... via {model_name}")
        result = generate_essay_details(essays[idx], client, model_slug)
        while len(existing) <= idx:
            existing.append(None)
        existing[idx] = result
        save_json_atomic(DETAILS_PATH, existing)
        print(f"[details] idx={idx}: wrote entry")
    print(f"[details] saved to {DETAILS_PATH}")


def _run_generation(args, mode_fn, mode_label):
    client = make_client()
    essays = load_essays()
    details = load_json(DETAILS_PATH, default=[])
    if len(details) != len(essays) or any(d is None for d in details):
        missing = [i for i, d in enumerate(details) if d is None]
        raise SystemExit(
            f"{DETAILS_PATH} is missing entries for indices {missing}; "
            f"run `python generate.py details` first."
        )

    metric_names = get_metric_names() if mode_fn is generate_essay_full else None

    models = load_models()
    if args.model:
        if args.model not in models:
            raise SystemExit(f"unknown model {args.model!r}; choices: {sorted(models)}")
        targets = [args.model]
    else:
        targets = sorted(models.keys())

    for model_name in targets:
        slug = models[model_name]
        count = find_target_count(model_name, len(essays))
        out_path = essays_path(model_name, count)
        partial = load_json(out_path, default=[])
        if not isinstance(partial, list):
            partial = []
        if len(partial) > len(essays):
            partial = partial[:len(essays)]

        indices = list(range(len(essays)) if args.essay_index is None else [args.essay_index])

        def produce(idx):
            if idx < len(partial) and partial[idx] is not None:
                return idx, None
            detail = details[idx]
            if mode_fn is generate_essay_naive:
                return idx, generate_essay_naive(detail, client, slug)
            corpus = same_author_essays_excluding(essays, idx)
            return idx, generate_essay_full(detail, corpus, metric_names, client, slug)

        def commit(idx, result):
            while len(partial) <= idx:
                partial.append(None)
            if result is not None:
                partial[idx] = result
                print(f"[{mode_label}] {model_name}#{count} idx={idx}: wrote essay")
            else:
                print(f"[{mode_label}] {model_name}#{count} idx={idx}: already present, skipping")
            save_json_atomic(out_path, partial)

        if args.workers and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(produce, i): i for i in indices}
                for fut in as_completed(futures):
                    idx, result = fut.result()
                    commit(idx, result)
        else:
            for idx in indices:
                idx_, result = produce(idx)
                commit(idx_, result)

        print(f"[{mode_label}] {model_name}#{count}: saved to {out_path}")


def cmd_naive(args):
    _run_generation(args, generate_essay_naive, "naive")


def cmd_full(args):
    _run_generation(args, generate_essay_full, "full")


def cmd_list_models(args):
    models = load_models()
    for name, slug in models.items():
        print(f"{name}\t{slug}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic essays mimicking essay authors via OpenRouter."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models defined in models.json and exit.",
    )
    sub = parser.add_subparsers(dest="command")

    p_details = sub.add_parser(
        "details", help="Generate essay_details.json (one assignment per essay)."
    )
    p_details.add_argument(
        "--model",
        default=None,
        help=f"Friendly model name from models.json (default: {DETAILS_DEFAULT_MODEL}).",
    )
    p_details.add_argument(
        "--essay-index", type=int, default=None, help="Only generate for this essay index."
    )

    p_naive = sub.add_parser(
        "naive", help="Generate essays_<model>_<count>.json with no extra context."
    )
    p_naive.add_argument("--model", default=None, help="Friendly model name from models.json.")
    p_naive.add_argument("--all", action="store_true", help="Run every model in models.json.")
    p_naive.add_argument("--essay-index", type=int, default=None, help="Only generate for this essay index.")
    p_naive.add_argument("--workers", type=int, default=1, help="Concurrent workers (default 1).")

    p_full = sub.add_parser(
        "full",
        help="Generate essays_<model>_<count>.json with full author corpus + metric list.",
    )
    p_full.add_argument("--model", default=None, help="Friendly model name from models.json.")
    p_full.add_argument("--all", action="store_true", help="Run every model in models.json.")
    p_full.add_argument("--essay-index", type=int, default=None, help="Only generate for this essay index.")
    p_full.add_argument("--workers", type=int, default=1, help="Concurrent workers (default 1).")

    args = parser.parse_args()

    if args.list_models:
        cmd_list_models(args)
        return

    if args.command == "details":
        cmd_details(args)
    elif args.command == "naive":
        if (args.model is None) == (not args.all):
            p_naive.error("specify exactly one of --model NAME or --all")
        cmd_naive(args)
    elif args.command == "full":
        if (args.model is None) == (not args.all):
            p_full.error("specify exactly one of --model NAME or --all")
        cmd_full(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

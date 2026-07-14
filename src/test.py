# Main way of running the test

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from analysis import (
    compare_metrics,
    get_all_metrics,
    plot_distribution,
    print_comparison,
)


def load_essays(path: str | None = None) -> list[dict]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "essays.json")
    with open(path) as f:
        return json.load(f)


def first_author_name(essays: list[dict]) -> str:
    return essays[0]["author"].replace("By ", "").strip()


def first_author_essays(essays: list[dict]) -> list[dict]:
    author = first_author_name(essays)
    return [e for e in essays if e["author"].replace("By ", "").strip() == author]


def main() -> None:
    essays = load_essays()
    author = first_author_name(essays)
    author_essays = first_author_essays(essays)
    print(f"First author: {author} ({len(author_essays)} essays)")

    text_a = author_essays[0]["body"] + "\n" + author_essays[1]["body"]
    text_b = author_essays[2]["body"]
    label_a = f"{author} essays 1+2"
    label_b = f"{author} essay 3"

    metrics_a = get_all_metrics(text_a)
    metrics_b = get_all_metrics(text_b)

    print(f"\n=== {label_a} vs {label_b} ===\n")
    results = compare_metrics(metrics_a, metrics_b, label_a, label_b)
    print_comparison(results, label_a, label_b)

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    plot_path = os.path.join(out_dir, "words_per_sentence_distribution.png")
    plot_distribution(
        metrics_a["words_per_sentence"] + metrics_b["words_per_sentence"],
        title="Words per Sentence (all three essays combined)",
        xlabel="Words per Sentence",
        save_path=plot_path,
    )
    print(f"\nSaved words-per-sentence histogram to {plot_path}")


if __name__ == "__main__":
    main()

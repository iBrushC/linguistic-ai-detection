import json
import os
import random

from datasets import load_dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESSAYS_PATH = os.path.join(REPO_ROOT, "essays.json")

DATASET_NAME = "Efstathios/guardian_authorship"
DATASET_CONFIG = "cross_topic_1"

AUTHOR_NAMES = {
    0: "Catherine Bennett",
    1: "George Monbiot",
    2: "Hugo Young",
    3: "Jonathan Freedland",
    4: "Martin Kettle",
}

TOPIC_NAMES = {
    0: "Politics",
    1: "Society",
    2: "UK",
    3: "World",
    4: "Books",
}

SEED = 42
ESSAYS_PER_AUTHOR = 5


def build_essays() -> list[dict]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)

    pool: list[dict] = []
    for split in ("train", "validation", "test"):
        if split in ds:
            pool.extend(ds[split])

    by_author: dict[int, list[dict]] = {aid: [] for aid in AUTHOR_NAMES}
    for row in pool:
        if row["author"] in by_author:
            by_author[row["author"]].append(row)

    rng = random.Random(SEED)
    essays: list[dict] = []
    for aid in sorted(AUTHOR_NAMES):
        rows = by_author[aid]
        rng.shuffle(rows)
        picked = rows[:ESSAYS_PER_AUTHOR]
        if len(picked) < ESSAYS_PER_AUTHOR:
            print(
                f"[warn] {AUTHOR_NAMES[aid]} only has {len(picked)} articles "
                f"in {DATASET_CONFIG}; using {len(picked)} instead of {ESSAYS_PER_AUTHOR}"
            )
        for row in picked:
            text = row["article"].strip()
            essays.append({
                "title": text.split("\n", 1)[0][:120],
                "author": AUTHOR_NAMES[aid],
                "intro": "",
                "body": text,
                "topic": TOPIC_NAMES[row["topic"]],
            })

    return essays


def topic_distribution(essays: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for e in essays:
        a = e["author"]
        out.setdefault(a, {})
        out[a][e["topic"]] = out[a].get(e["topic"], 0) + 1
    return out


if __name__ == "__main__":
    essays = build_essays()
    with open(ESSAYS_PATH, "w", encoding="utf-8") as f:
        json.dump(essays, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(essays)} essays across {len(AUTHOR_NAMES)} authors to {ESSAYS_PATH}")
    print("\nPer-author topic distribution:")
    dist = topic_distribution(essays)
    for author in sorted(dist):
        topics = ", ".join(f"{t}={c}" for t, c in sorted(dist[author].items()))
        print(f"  {author:<22} {topics}")
"""Topic-leakage diagnostic.

If a content-only classifier approaches the function-word pipeline's
accuracy, the function-word signal is being driven by subject matter
rather than style. We surface this as a prominent warning.

Two classifiers are fit on the same chunked corpus:

* **content-only** - open-class POS tags (nouns/verbs/adjectives/adverbs)
  minus NLTK English stopwords
* **MFW (function-leaning)** - the same MFW feature space used by the
  attribution pipeline

Both use simple logistic regression with 5-fold stratified cross
validation. The content-only accuracy is the warning trigger.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.features import content_words


DEFAULT_WARNING_GAP = 0.05
DEFAULT_CV = 5
RANDOM_STATE = 1729


def _content_corpus(chunks: Iterable[dict]) -> list[str]:
    return [" ".join(content_words(c["text"])) for c in chunks]


def _mfw_corpus(chunks: Iterable[dict], top_k: int | None = None) -> list[str]:
    """Lexical-words-only (lowercased words). Used as a proxy MFW corpus."""
    if top_k is None:
        return [c["text"].lower() for c in chunks]
    return [" ".join(c["text"].lower().split()[:top_k]) for c in chunks]


def _evaluate(texts: list[str], labels: np.ndarray, cv: int = DEFAULT_CV) -> dict:
    if len(set(labels)) < 2:
        return {"accuracy": float("nan"), "n_folds_used": 0}
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True,
                          ngram_range=(1, 1))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000,
                             random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy", n_jobs=1)
    return {
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "per_fold": scores.tolist(),
        "n_features": int(X.shape[1]),
        "n_chunks": int(X.shape[0]),
    }


def topic_leakage_check(
    chunks: list[dict],
    cv: int = DEFAULT_CV,
    warning_gap: float = DEFAULT_WARNING_GAP,
) -> dict:
    """Compute content-only and MFW classification accuracies.

    ``chunks`` is a list of dicts each with ``text`` and ``author``.
    Returns a report dict including ``warning`` (bool) and console
    ``message`` strings.
    """
    labels = np.array([c["author"] for c in chunks])

    content_texts = _content_corpus(chunks)
    content_metrics = _evaluate(content_texts, labels, cv=cv)

    mfw_texts = _mfw_corpus(chunks)
    mfw_metrics = _evaluate(mfw_texts, labels, cv=cv)

    chance = 1.0 / len(set(labels))
    gap = mfw_metrics["accuracy"] - content_metrics["accuracy"]
    warn = (content_metrics["accuracy"] - chance) > warning_gap and gap < warning_gap

    if warn:
        message = (
            "[TOPIC LEAK WARNING] "
            f"content-word classifier accuracy ({content_metrics['accuracy']:.3f}) "
            f"is within {warning_gap:.2f} of the MFW classifier "
            f"({mfw_metrics['accuracy']:.3f}); the function-word signal may "
            "be tracking subject matter rather than style."
        )
    else:
        message = (
            "[topic leakage] content-word accuracy = "
            f"{content_metrics['accuracy']:.3f}; MFW accuracy = "
            f"{mfw_metrics['accuracy']:.3f}; chance = {chance:.3f}."
        )

    per_author_counts = dict(sorted(
        ((a, sum(1 for c in chunks if c['author'] == a))
         for a in set(c['author'] for c in chunks))
    ))

    return {
        "warning": bool(warn),
        "message": message,
        "n_chunks": len(chunks),
        "per_author_chunk_counts": per_author_counts,
        "chance_accuracy": float(chance),
        "cv_folds": cv,
        "warning_gap": warning_gap,
        "content_only": content_metrics,
        "mfw": mfw_metrics,
        "accuracy_gap_mfw_minus_content": float(gap),
    }


__all__ = ["topic_leakage_check", "DEFAULT_WARNING_GAP"]

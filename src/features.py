"""Feature extractors for stylometric Delta-style distances.

Two extractors are provided behind a small protocol:

* :class:`MFWExtractor` - relative frequencies of the top-N most frequent
  word types in the combined training+reference corpus. Function words are
  intentionally kept; their frequencies carry most of the author signal.

* :class:`CharNGramExtractor` - character 3- and 4-grams, tf-idf weighted,
  preserving whitespace and punctuation.

Each extractor exposes ``fit_transform`` over the union corpus and
``transform`` for individual texts at evaluation time. The output is a
``(n_items, n_features)`` sparse matrix.
"""

from __future__ import annotations

import abc
import math
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_WORD_RE = re.compile(r"\w+")


def _text_of(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text") or item.get("body") or item.get("token_text") or ""
    if isinstance(item, (list, tuple)):
        return " ".join(item)
    raise TypeError(f"cannot extract text from {type(item).__name__}")


class FeatureExtractor(abc.ABC):
    name: str
    needs_zscoring: bool = True

    @abc.abstractmethod
    def fit(self, corpus: Sequence) -> None: ...

    @abc.abstractmethod
    def transform(self, items: Sequence): ...

    def fit_transform(self, items: Sequence):
        self.fit(items)
        return self.transform(items)

    @abc.abstractmethod
    def get_feature_names(self) -> list[str]: ...

    @property
    @abc.abstractmethod
    def reference_stats(self) -> dict | None:
        """Per-feature mu, sigma over the reference corpus, or None."""


# ---------------------------------------------------------------------------
# MFW
# ---------------------------------------------------------------------------


class MFWExtractor(FeatureExtractor):
    """Top-N most frequent word types, relative frequencies."""

    name = "mfw"
    needs_zscoring = True

    def __init__(self, n: int = 500, lowercase: bool = True):
        if not 100 <= n <= 1000:
            raise ValueError(f"mfw N must be in [100, 1000]; got {n}")
        self.n = n
        self.lowercase = lowercase
        self.vocab_: list[str] = []
        self._vocab_index: dict[str, int] = {}
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    @staticmethod
    def _tokenize(text: str, lowercase: bool) -> list[str]:
        words = _WORD_RE.findall(text)
        if lowercase:
            words = [w.lower() for w in words]
        return words

    def _count(self, text: str) -> Counter:
        return Counter(self._tokenize(text, self.lowercase))

    def fit(self, corpus: Sequence) -> None:
        counts: Counter = Counter()
        for item in corpus:
            counts.update(self._count(_text_of(item)))

        types = [w for w, _ in counts.most_common(self.n)]
        self.vocab_ = types
        self._vocab_index = {w: i for i, w in enumerate(types)}

        ref_vectors = self._vectorise_many([_text_of(it) for it in corpus])
        if ref_vectors.size == 0:
            self._mu = np.zeros(len(self.vocab_), dtype=np.float64)
            self._sigma = np.zeros(len(self.vocab_), dtype=np.float64)
        else:
            self._mu = ref_vectors.mean(axis=0)
            self._sigma = ref_vectors.std(axis=0, ddof=0)

    def _vectorise_one(self, text: str):
        total = 0
        row = np.zeros(len(self.vocab_), dtype=np.float64)
        if self._vocab_index:
            counts = self._count(text)
            total = sum(counts.values())
            if total > 0:
                for tok, c in counts.items():
                    j = self._vocab_index.get(tok)
                    if j is not None:
                        row[j] = c / total
        return row

    def _vectorise_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.vocab_)), dtype=np.float64)
        rows = [self._vectorise_one(t) for t in texts]
        return np.vstack(rows)

    def transform(self, items: Sequence):
        texts = [_text_of(it) for it in items]
        return self._vectorise_many(texts)

    def get_feature_names(self) -> list[str]:
        return list(self.vocab_)

    @property
    def reference_stats(self) -> dict | None:
        if self._mu is None:
            return None
        return {
            "mu": self._mu.tolist(),
            "sigma": self._sigma.tolist(),
        }




# ---------------------------------------------------------------------------
# Char n-grams
# ---------------------------------------------------------------------------


class CharNGramExtractor(FeatureExtractor):
    """Character 3-grams and 4-grams, tf-idf weighted."""

    name = "char_ngram"
    needs_zscoring = False

    def __init__(self, ngram_range: tuple[int, int] = (3, 4), min_df: int = 2,
                 sublinear_tf: bool = True, max_features: int | None = None):
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features
        self.vectorizer_: TfidfVectorizer | None = None
        self._cached_vocab_size: int = 0

    def _texts(self, items: Sequence) -> list[str]:
        return [_text_of(it) for it in items]

    def fit(self, corpus: Sequence) -> None:
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            lowercase=False,
        )
        self.vectorizer_.fit(self._texts(corpus))
        self._cached_vocab_size = len(self.vectorizer_.vocabulary_)

    def transform(self, items: Sequence):
        if self.vectorizer_ is None:
            raise RuntimeError("CharNGramExtractor.transform called before fit")
        return self.vectorizer_.transform(self._texts(items))

    def fit_transform(self, items: Sequence):
        if self.vectorizer_ is None:
            self.vectorizer_ = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
                max_features=self.max_features,
                lowercase=False,
            )
            return self.vectorizer_.fit_transform(self._texts(items))
        return self.transform(items)

    def get_feature_names(self) -> list[str]:
        if self.vectorizer_ is None:
            return []
        return [k for k, _ in sorted(self.vectorizer_.vocabulary_.items(),
                                      key=lambda kv: kv[1])]

    @property
    def reference_stats(self) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# Content-word extractor for topic-leakage diagnostic
# ---------------------------------------------------------------------------


_CONTENT_POS = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
}


def content_words(text: str) -> list[str]:
    """Open-class content words, lowercased, stopwords removed."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    try:
        stop = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop = set(stopwords.words("english"))

    out: list[str] = []
    try:
        tokens = word_tokenize(text)
        tags = nltk.pos_tag(tokens)
    except LookupError:
        for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
            try:
                nltk.data.find(f"taggers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        tokens = word_tokenize(text)
        tags = nltk.pos_tag(tokens)

    for w, t in tags:
        wl = w.lower()
        if not wl.isalpha():
            continue
        if t not in _CONTENT_POS:
            continue
        if wl in stop:
            continue
        out.append(wl)
    return out


__all__ = [
    "FeatureExtractor",
    "MFWExtractor",
    "CharNGramExtractor",
    "content_words",
]

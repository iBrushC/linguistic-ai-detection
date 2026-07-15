# Functions for styometric markers within text

"""
- Lexical markers
    - Word length distribution
    - Sentence length distribution
    - Words per sentence distribution
    - Type/token ratio (TTR and CTTR)
- Morphological markers
    - Adjective
    - Adposition
    - Adverb
    - Auxiliary
    - Coordinating conjunction
    - Subordinating conjunction
    - Noun
    - Particle
    - Pronoun
    - Verb
- Syntactical markers
    - Adverbial modifier
    - Adjectival modifier
    - Conjunct
    - Determiner
    - Direct object
    - Nominal subject
    - Object of preopsition
    - Root
- Structural markers
    - Tricolon
    - Cleft frequency
    - Normalization rate
    - Existential and Extraposition rate
    - Anadiplosis
    - Conjunctions per coordinated series
    - Connective density and type mix (additive / adversative / causal / temporal)
    - Segments per sentence
"""

import re
import nltk
import numpy as np
import pandas
import spacy
from scipy.stats import ks_2samp, mannwhitneyu, brunnermunzel

# If not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def get_sentence_lengths(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    return [len(s) for s in sentences]

def get_words_per_sentence(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    words_per_sentence = []
    for s in sentences:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(s)
        words_per_sentence.append(len(words))

    return words_per_sentence

def get_word_types_per_sentence(text: str) -> dict:
    sentences = nltk.tokenize.sent_tokenize(text)
    sentence_counts = []
    all_tags = set()

    for s in sentences:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(s)
        counts = {}
        for _, pos_tag in nltk.pos_tag(words):
            counts[pos_tag] = counts.get(pos_tag, 0) + 1
        sentence_counts.append(counts)
        all_tags.update(counts)

    return {
        tag: [counts.get(tag, 0) for counts in sentence_counts]
        for tag in sorted(all_tags)
    }

def get_word_lengths(text: str) -> list[int]:
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    return [len(w) for w in words]

# --- Lexical: TTR / CTTR (per-sentence) ---

def get_ttr_per_sentence(text: str) -> list[float]:
    sentences = nltk.tokenize.sent_tokenize(text)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    out = []
    for s in sentences:
        words = [w.lower() for w in tokenizer.tokenize(s)]
        if not words:
            out.append(0.0)
            continue
        out.append(len(set(words)) / len(words))
    return out

def get_cttr_per_sentence(text: str) -> list[float]:
    sentences = nltk.tokenize.sent_tokenize(text)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    out = []
    for s in sentences:
        words = [w.lower() for w in tokenizer.tokenize(s)]
        n = len(words)
        if n < 2:
            out.append(0.0)
            continue
        out.append(len(set(words)) / np.sqrt(2 * n))
    return out

# --- Syntactical markers (spaCy dependency labels per sentence) ---

_SYNTACTIC_DEPS = {
    "adverbial_modifier": "advmod",
    "adjectival_modifier": "amod",
    "conjunct": "conj",
    "determiner": "det",
    "direct_object": "dobj",
    "nominal_subject": "nsubj",
    "object_of_preposition": "pobj",
    "root": "ROOT",
}

def get_syntactic_markers(text: str) -> dict:
    nlp = _get_nlp()
    sentences = nltk.tokenize.sent_tokenize(text)
    counts = {key: [] for key in _SYNTACTIC_DEPS}
    for doc in nlp.pipe(sentences):
        sentence_counts = {key: 0 for key in _SYNTACTIC_DEPS}
        for tok in doc:
            for key, dep in _SYNTACTIC_DEPS.items():
                if tok.dep_ == dep:
                    sentence_counts[key] += 1
        for key in _SYNTACTIC_DEPS:
            counts[key].append(sentence_counts[key])
    return counts

# --- Structural markers ---

def get_tricolon_counts(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    out = []
    pattern = re.compile(
        r"\b([^\s,][^,]*?),\s+([^\s,][^,]*?),\s+(?:and\s+)?([^\s,][^,\.]*?)(?:[\.\;])",
        re.IGNORECASE,
    )
    for s in sentences:
        out.append(len(pattern.findall(s)))
    return out

def get_cleft_counts(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    out = []
    cleft_re = re.compile(
        r"\bit\s+(?:is|was|be|been|being)\s+[^,\.]+?\s+(?:that|who|which)\b",
        re.IGNORECASE,
    )
    what_re = re.compile(
        r"\bwhat\s+[^,\.]+?\s+(?:is|was|are|were)\s+[^,\.]+",
        re.IGNORECASE,
    )
    for s in sentences:
        out.append(1 if (cleft_re.search(s) or what_re.search(s)) else 0)
    return out

def get_normalization_counts(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    out = []
    there_re = re.compile(r"\bthere\s+(?:is|are|was|were|seems?|appears?|exist(?:s|ed)?)\b", re.IGNORECASE)
    for s in sentences:
        out.append(1 if there_re.search(s) else 0)
    return out

def get_existential_extraposition_counts(text: str) -> list[int]:
    existential_re = re.compile(r"\bthere\s+(?:is|are|was|were)\b", re.IGNORECASE)
    extraposition_re = re.compile(r"\bit\s+(?:is|was)\s+[^,\.]+?\s+(?:that|who|which)\b", re.IGNORECASE)
    sentences = nltk.tokenize.sent_tokenize(text)
    out = []
    for s in sentences:
        if existential_re.search(s) or extraposition_re.search(s):
            out.append(1)
        else:
            out.append(0)
    return out

def get_anadiplosis_counts(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    if not sentences:
        return []
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    last_words = []
    for s in sentences:
        words = tokenizer.tokenize(s)
        tail = [w.lower() for w in words[-3:]] if words else []
        last_words.append(tail)
    out = [0]
    for i in range(1, len(sentences)):
        prev_tail = last_words[i - 1]
        curr_words = [w.lower() for w in tokenizer.tokenize(sentences[i])]
        if not prev_tail or not curr_words:
            out.append(0)
            continue
        match = 1 if any(w in curr_words[:3] for w in prev_tail) else 0
        out.append(match)
    return out

def get_conjunctions_per_series(text: str) -> list[int]:
    sentences = nltk.tokenize.sent_tokenize(text)
    out = []
    list_re = re.compile(r",\s+(?:and|or)\s+[^,]", re.IGNORECASE)
    for s in sentences:
        items = [seg.strip() for seg in s.split(",") if seg.strip()]
        if len(items) < 3:
            out.append(0)
            continue
        out.append(len(list_re.findall(s)))
    return out

_CONNECTIVES = {
    "additive": {"and", "also", "moreover", "furthermore", "additionally", "besides"},
    "adversative": {"but", "however", "yet", "nevertheless", "nonetheless", "although", "though"},
    "causal": {"because", "therefore", "thus", "hence", "so", "consequently", "since"},
    "temporal": {"then", "after", "before", "while", "when", "until", "since", "meanwhile", "previously"},
}

def get_connective_density(text: str) -> dict:
    sentences = nltk.tokenize.sent_tokenize(text)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    out = {key: [] for key in _CONNECTIVES}
    for s in sentences:
        words = [w.lower() for w in tokenizer.tokenize(s)]
        per_sent = {key: 0 for key in _CONNECTIVES}
        for w in words:
            for key, vocab in _CONNECTIVES.items():
                if w in vocab:
                    per_sent[key] += 1
        for key in _CONNECTIVES:
            out[key].append(per_sent[key])
    return out

def get_segments_per_sentence(text: str) -> list[int]:
    nlp = _get_nlp()
    sentences = nltk.tokenize.sent_tokenize(text)
    out = []
    for doc in nlp.pipe(sentences):
        n_clauses = sum(1 for tok in doc if tok.dep_ in ("ROOT", "conj"))
        out.append(max(1, n_clauses))
    return out
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
    - Active vs passive voice counts
    - Cleft frequency
    - Normalization rate
    - Existential and Extraposition rate
    - Anadiplosis
    -  Conjunctions per coordinated series
    - Connective density and type mix (additive / adversative / causal / temporal)
    - Segments per sentence
"""

import nltk
import numpy as np
import pandas
from scipy.stats import ks_2samp, mannwhitneyu, brunnermunzel

# If not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


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
    type_counts = {}

    for s in sentences:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(s)
        tag_pairs = nltk.pos_tag(words)
        sentence_type_counts = {}
        for (word, pos_tag) in tag_pairs:
            if (pos_tag in sentence_type_counts.keys()):
                sentence_type_counts[pos_tag] += 1
            else:
                sentence_type_counts[pos_tag] = 1

        for key in sentence_type_counts.keys():
            if (key in list(type_counts.keys())):
                type_counts[key].append(sentence_type_counts[key])
            else:
                type_counts[key] = [sentence_type_counts[key]]

    return type_counts

def get_word_lengths(text: str) -> list[int]:
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    return [len(w) for w in words]
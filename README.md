# Linguistic AI Detection

Stylometric authorship attribution aimed at identifying individual human
authors in order to detect AI-generated text by inversion. The pipeline
measures effect-size distances between text chunks under a held-out
reference, scores them with self-vs-cross AUC, and reports a permutation
p-value.

## What

We **do not** use significance tests (KS, MANOVA, Brunner-Munzel) as
similarity scores. They scale with sample size, not stylistic difference:
the same author looks "significantly different" given enough tokens, and
different authors look "not significant" given few. Instead, we compute
effect-size distances between equally-sized chunks of text and rank them.

Three distances run on two feature spaces:

| Metric | Feature space | Notes |
| --- | --- | --- |
| Cosine Delta (target) | MFW relative frequencies | z-score against the held-out reference, then cosine distance |
| Burrows's Delta | MFW relative frequencies | z-score against the held-out reference, then Manhattan |
| Char n-gram cosine | 3- and 4-grams (tf-idf) | second baseline |

## Dataset

Two disjoint slices from `Efstathios/guardian_authorship`:

* **test essays** (`essays.json`): all authors 0–4 in `cross_topic_1`
  (train + validation + test splits). 135 articles.
* **reference corpus** (`reference_corpus.json`): the same five authors
  in `cross_genre_2`, `cross_genre_3`, `cross_genre_4`. 456 articles.
  Held out of the test set; z-score statistics come from this slice.

The five target authors are:

* Catherine Bennett
* George Monbiot
* Hugo Young
* Jonathan Freedland
* Martin Kettle

## Pipeline

```
python -m src.dataset                  # rebuild essays.json + reference_corpus.json
python -m src.chunking --chunk-size 1000   # produce chunked test corpus
python -m src.run_pipeline             # default config, 250 permutations
python -m src.run_pipeline --permutations 1000   # full significance test
python -m src.run_pipeline --distance cosine_delta   # one metric at a time
python -m pytest tests                 # unit tests
```

Chunking rejects shorter-than-`min_fill_ratio` tails and aborts if any
author has fewer than `min_chunks_per_author` chunks. The loader prints
per-author article and token counts **before** any modeling so corpus
imbalance is visible up front.

## Outputs (`src/plots/`)

* `corpus_report.json` — chunk counts, raw doc lengths, per-author tokens
* `vocab_<feature>.json` — feature-extractor vocabulary head
* `distances/<metric>.npy` — the full distance matrix
* `distances_overlap_<metric>.png` — histogram of self vs cross distances
* `evaluation_<metric>.json` — `{auc, n_pairs, perm_pvalue, ...}`
* `topic_leakage.json` — content-only vs MFW accuracy + warning flag
* `run_summary.json` — combined summary

## Topic leakage

The diagnostic trains a content-only author classifier (open-class POS,
stopwords removed) and compares its accuracy to the MFW/function-word
pipeline. If content words do as well, the metric is measuring subject
matter rather than style — the report surfaces this as
`[TOPIC LEAK WARNING]`.

## Tests

```
python -m pytest tests
```

Boundary cases for chunking, z-score against a held-out reference,
distance-matrix symmetry, AUC monotonicity, permutation p-value,
end-to-end smoke test on a synthetic corpus.

## Configuration

`src/configs/default.json` carries every knob: chunk size, MFW N,
reference corpus path, feature set, distance set, permutation count,
output directory. CLI flags override individual keys.

## OpenRouter cross-validation experiment

Tests whether Cosine Delta can flag AI text masquerading as a target
human author. Picks five authors and five essays per author (seed
1729), generates one shared writing assignment per essay with
`glm-5.2`, recreates each essay with the same model using the other
four natural essays for stylistic context, then runs five
leave-one-out folds per author.

```
set OPENROUTER_API_KEY=...
python -m src.experiment select --out-dir generated
python -m src.generate details --model glm-5.2 --use-selector generated/experiment_manifest.json
python -m src.generate recreate --model glm-5.2 --use-selector generated/experiment_manifest.json
python -m src.experiment score --manifest generated/experiment_manifest.json --out-dir src/plots/experiment_first
```

For every fold the scoring fits MFW vocabulary on natural chunks and
reference mu/sigma on `reference_corpus.json` only; AI chunks never
participate in fitting. The mean Cosine Delta from each target's
chunks to the four-essay corpus is computed for both the natural and
the AI-recreated essays. Outputs:

* `generated/experiment_manifest.json` -- deterministic selection.
* `generated/essay_details.json` -- shared assignments.
* `generated/essays_glm-5.2_recreate.json` -- recreations.
* `src/plots/experiment_<run>/folds.json` -- per-fold distances.
* `src/plots/experiment_<run>/summary.json` -- per-author and overall
  mean/SD.
* `src/plots/experiment_<run>/folds_cosine_delta.png` -- grouped
  natural vs AI bar chart with SD error bars.

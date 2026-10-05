# The `laya` backend

An ensemble backend that re-scores subject suggestions using the Laya
decision model (<https://laya.convaiinnovations.com>, Hugging Face:
`convaiinnovations/laya`, PyPI package `laya`). Laya is an open-weight
"System 1" decision model: given a text and a question it returns
calibrated probabilities in a single forward pass, without generating
any text.

This is the single documentation home for the backend: parameter
guidance and the tuning workflow. Measurements in this document come
from tuning a YAKE-sourced development corpus and are in-sample;
re-measure on your own corpus before relying on any figure.

## How it works

For each document the backend sends Laya **one request** whose
questions are chosen by `laya_score_mode`:

- `choice` — a single **choice question** (*"Which subject does this
  text reference?"*) whose options are the candidate labels. Laya
  compares the candidates against each other and returns a probability
  for every option.
- `noul` — one **yes/no proposition** per candidate (*"This document is
  about {label}."*), each scored independently.
- `choice,noul` (equivalently `noul,choice`) — both of the above in the
  same request.

The scoring stage then blends the selected answers with the source
scores following the CLM backend's model:

```
sources ──> weighted merge (top-`limit` candidates)
              │
              ▼
       one Laya request per document
       (questions per `laya_score_mode`)
              │
              ▼
       per-document min-max normalization of the Laya
       scores (and of the source scores, unless
       blend-source = raw)
              │
              ▼
       score = blend-alpha · source
             + (1 − blend-alpha) · norm(laya)
              │
              ▼
       keep the top-`limit` candidates, return
```

The questions asked — at suggest time and during `annif hyperopt` —
are those of the configured mode. A `choice,noul` configuration makes
the hyperopt caching pass answer both question kinds, so its trials
compare all three modes without new inference; a single-mode
configuration makes the pass much cheaper (with `choice`, back to one
question per document) and the trials tune `blend-alpha` at that
mode.

### Scoring rules

- Min-max normalization is per document and per score set, and always
  applies to the Laya scores. With the default `blend-source =
  normalized` it applies to the source scores too, so a document with
  a single candidate — where both sets are degenerate — always scores
  0.5. With `blend-source = raw` the source score keeps its absolute
  value, so a one-candidate document scores
  `blend-alpha × source + (1 − blend-alpha) × 0.5`.
- In the combined mode the Laya value of a candidate is the **average
  of the answers that exist**: if one answer set is entirely missing,
  scoring falls back to the other signal; per-candidate, a single
  available answer is used as-is.
- A candidate with no answer at all counts as 0.0 — *not* as its
  source score. The weakest candidate of a document can therefore
  blend to exactly 0.0 and be dropped from the results. Misses are
  visible in DEBUG output: `missing Laya score for subject(s) …,
  counting as 0.0`.
- If the whole Laya request fails, every candidate of that document
  keeps its original source score instead. This fallback is
  deliberately different from a per-candidate miss, which counts as
  0.0.
- The blend can reorder candidates; the output is re-sorted by blended
  score, and the output `limit` keeps the top candidates by blended
  score, not by the source ranking they arrived in.

The models are never trained — the pipeline is fully zero-shot.

## Parameters

| parameter | meaning | default |
|---|---|---|
| `sources` | source projects, as in the plain ensemble | required |
| `limit` | candidate pool: how many merged candidates are compared (and the cap on returned suggestions) | 20 |
| `laya_score_mode` | which questions Laya is asked and which answers score the candidates: `choice`, `noul`, or `choice,noul` | `choice` |
| `blend-alpha` | weight of the source score in the blend (normalized or raw per `blend-source`) | 0.85 |
| `blend-source` | how the source scores enter the blend: min-max normalized per document (`normalized`) or as-is (`raw`) | `normalized` |
| `question_template` | the choice question's instruction text | "Which subject does this document reference?" |
| `instruction` | the per-candidate proposition text; `{label}` is required | "This document is about {label}." |
| `laya_model` | checkpoint | `multilingual` (empty value restores language-based auto-routing) |
| `laya_max_len` / `laya_head_max_len` | token budgets (see below) | the checkpoint's own values |
| `laya_input_chars` | cap on the raw text sent to Laya (tokenizer savings) | 0 = full text |

### `limit` — the candidate pool

Keep it at or below ~20: Laya's own documentation warns that choice
accuracy degrades as the options outgrow the question token budget,
and in the modes that use propositions every candidate adds one to the
request, so very large pools inflate the question text. Keep it well
above 2: measured on a development corpus, a pool of only the top-2
candidates capped gold recovery at 0.50, versus 0.80 at depth 20 —
the ceiling collapses before Laya even gets to rank (this figure
measures the source candidate pool, not the scoring). The returned set
is the top-`limit` candidates by blended score; callers can shrink it
per request (`-t` on the CLI, `threshold` in the REST API), which
filters the same blended scores from outside the backend, as elsewhere
in Annif.

### `blend-alpha` and `laya_score_mode` — the blend

```
final score = blend-alpha × source + (1 − blend-alpha) × norm(laya)
```

`blend-source` picks how the source scores enter the blend:
`normalized` (the default, CLM-style) min-max normalizes them per
document like the Laya scores, so the blend weighs two *relative*
rankings; `raw` uses them as-is, keeping their absolute confidence.
The raw variant matters when the source's score scale itself carries
signal — for example YAKE, whose top keyword of nearly any document
lands near saliency 1.0, so an absolute threshold on the blend can mean
"the extractor is confident, unless the verifier contradicts it";
per-document normalization discards exactly that.

`blend-alpha = 1.0` keeps the source ranking (normalized or raw
according to `blend-source`). `blend-alpha = 0.0` is pure replacement:
the output ranking is the normalized Laya ranking of the selected
mode, and the weakest Laya candidate scores 0.0. The 0.85 default is
CLM's: trust the source as the primary ranker and let the zero-shot
verifier adjust about 15%. Tune it per project with `annif hyperopt`,
which also tunes `blend-source` — but note that hyperopt optimizes an
unthresholded metric; the best *thresholded* operating point can
prefer a very different pair (see the tuning workflow).

`laya_score_mode` selects both the questions and the answers used for
scoring: `choice` compares the candidates against each other in one
question; `noul` scores each candidate independently; `choice,noul`
asks both and averages the available answers per candidate.

### Output sizing

There is no backend-level threshold: the output is the top-`limit`
candidates by blended score, and candidates that blend to exactly 0.0
are dropped. Apply a threshold per call instead (`-t` on the CLI,
`threshold` in the REST API) — it filters the same blended scores, so
the effect is identical to an internal threshold. A `threshold` line
in the project config is ignored by this backend.

With `blend-source = raw` and a high `blend-alpha` the threshold has a
structural meaning: at `blend-alpha = 0.75`, a threshold of 0.75 admits
only candidates whose source score is near-perfect — or that Laya's
answers rank first. "The extractor is confident, unless the verifier
contradicts it." Just below that point the output floods: extractors
like YAKE produce a band of near-top candidates whose scores cluster
just under the top one, and they are mostly wrong. Measured on a
YAKE-sourced development corpus (in-sample), raising the threshold
from 0.70 to 0.75 shrank the output from ~7.6 to ~1.8 suggestions per
document while precision rose from 0.19 to 0.53 — the threshold is a
cliff, not a dial, so probe around the knee rather than assuming a
smooth landscape.

### `instruction` — the proposition text

The `{label}` placeholder is required; a proposition without it is a
configuration error. The default is
*"This document is about {label}."* Keep propositions short: each one
is part of the question text of its own question sequence (see the
token budgets below).

## Token budgets

Laya encodes each question as one sequence
`[document text | question + options]` of at most `laya_max_len` tokens,
of which `laya_head_max_len` are reserved for the question. The model
reads `laya_max_len − laya_head_max_len` tokens of the document;
everything past that is sliced away inside Laya.

Verified from the shipped checkpoint configurations:

| checkpoint | `max_len` | `head_max_len` | reads | encoder ceiling |
|---|---|---|---|---|
| english (root) | 512 | 192 | 320 tokens (~1,300 chars) | 8192 |
| multilingual | 1024 | 256 | 768 tokens (~3,000 chars) | 8192 |

Both encoders accept up to 8,192 tokens, but **only when `laya_max_len`
is explicitly raised** — the 8,192 figure is a capability, not a
default. Doubling `max_len` doubles the compute per question.

Neither budget is set by default: the checkpoint's own config applies —
with the default `multilingual` checkpoint that means 1024/256, reading
~3,000 characters. Note that the question side of each sequence holds
one proposition in the noul modes, and the choice question holds one
option per candidate: keep `laya_head_max_len` comfortably above the
space the questions need, or lower `limit`.

`laya_input_chars` complements the budgets: it truncates the raw text
*before* it is sent to Laya, so the tokenizer — which would otherwise
process the whole document on every call even though the model keeps
only its state budget of it — is not fed text that can never be read. A
value slightly above `(laya_max_len − laya_head_max_len) × 4`
characters per token is result-neutral for English documents, and must
be raised if the budgets are raised.

## Tuning workflow

Two kinds of decision, two tools:

| decision | tool |
|---|---|
| ranking config (`blend-alpha`, `blend-source`, `laya_score_mode`) | `annif hyperopt` — trials re-score the answers cached by a single inference pass, so hundreds cost nothing |
| operating point (per-call `limit`, `threshold`) | `annif eval` sweeps (or the `annif optimize` grid) at the candidate blends |

The split matters because the two optima need not agree: hyperopt
maximizes an *unthresholded* metric over the full candidate pool, which
rewards ranking; a thresholded operating point rewards confidence, and
can prefer a very different blend. Paste the hyperopt lines into the
config only when you serve unthresholded output; for a thresholded
profile, sweep before deciding.

On a validation corpus distinct from your evaluation corpus — values
tuned on the evaluation set overfit, which was observed repeatedly
during development:

1. **Tune the ranking config** with `annif hyperopt`:

   ```bash
   annif hyperopt <project> <corpus> -T 200 -m "F1 score (doc avg)"
   ```

   Any metric works, including the default NDCG. The recommendation
   covers `blend-alpha` and `blend-source`, plus `laya_score_mode` when
   the configured mode is `choice,noul` (the cached pass then answers
   both question kinds, so the trials compare all three modes).

2. **Find the operating point** by sweeping the per-call `limit` and
   `threshold` at the candidate blends. In `choice` mode each eval is
   a single Laya pass over the corpus, so a sweep is cheap:

   ```bash
   for t in 0.70 0.75 0.80 0.85; do
       annif eval <project> <corpus> -l 20 -t $t \
           -b laya.blend-alpha=0.75 -b laya.blend-source=raw
   done
   ```

   Sweep a couple of blends as well — the thresholded optimum can
   prefer a different `blend-alpha`/`blend-source` than hyperopt's
   unthresholded one (for a YAKE-like source it does). Remember the
   cliff: once a threshold looks promising, probe the values just
   around it.

3. **Confirm on a held-out corpus**: set the winning blend in the
   project config, apply the winning `-l`/`-t` per call, and evaluate
   once on data that no tuning step has seen. That is the only number
   to report.

## Guidance from development

The backend's value is proportional to the weakness of its source. For
an already-strong source no variant tested beat the source alone; use
a trained source when one exists, and Laya where one does not.

Findings from tuning a YAKE-sourced project on a 5,664-document corpus
(in-sample, under the current scoring model):

- **`choice` beat `noul` and `choice,noul` for ranking.** With roughly
  one gold subject per document, the comparative question ("which of
  these?") fits the problem better than independent per-candidate
  propositions.
- **Two optima, two blends.** Unthresholded ranking was best at
  `blend-alpha ≈ 0.07` with `blend-source = normalized` — nearly pure
  Laya ranking. The thresholded operating point was best at
  `blend-alpha = 0.75` with `blend-source = raw` and `-t 0.75` — the
  "extractor confident unless the verifier contradicts" regime. On the
  tuning corpus it slightly exceeded the historical tuned profile of
  the previous scoring model (doc-avg F1 0.573 vs 0.570, both
  in-sample), and it held doc-avg F1 0.563 on the held-out evaluation
  corpus — about a point of tuning-corpus optimism, and no sign of
  overfitting. Neither blend wins both regimes; tune for the regime you
  deploy.
- **Ranking strength concentrates at the top.** Precision@1 was
  ~0.53–0.55 at every operating point, while the second suggestion was
  right only ~15% of the time: the pipeline is an excellent
  single-subject extractor, and depth beyond the first hit needs a
  corpus that actually assigns multiple subjects per document.

Treat these as direction, not measurement: re-measure on your own
corpus before making decisions based on them.

## Operation notes

- **Not trainable.** Training the laya backend raises
  `NotSupportedException`; train the source projects instead.
- **Hyperopt internals.** `annif hyperopt` runs the source merge once
  and asks Laya the question kinds of the configured mode in a single
  pass per document (`_prepare`), caching their answers for every
  candidate. Each trial then re-scores the cache with its own
  `blend-alpha` and `blend-source` — and `laya_score_mode`, when the
  cache covers more than one mode — exactly like the backend does at
  suggest time, so the objective mirrors the backend's output path and
  the trials cost no inference.
- **Progress logging.** At suggest time one INFO line per batch:
  `Laya: 32 doc(s), 155 candidate(s), 1.30s`. During `annif hyperopt`
  the caching pass reports progress per document batch:
  `Laya: cached 160/5664 doc(s) (2.8%), 9327 question(s), 48.2s,
  ~27 min left`. Each failed request logs a warning. At DEBUG
  (`annif -v DEBUG`): one line per document with its question count
  and position in the batch, and one line per document listing the
  candidates whose Laya answer is missing.
- **Performance.** Cost is one forward pass per document regardless of
  the candidate count; the mode decides how much question text that
  pass carries (one choice question, one proposition per candidate,
  or both).

# The `laya` backend

An ensemble backend that re-scores subject suggestions using the Laya
decision model (<https://laya.convaiinnovations.com>, Hugging Face:
`convaiinnovations/laya`, PyPI package `laya`). Laya is an open-weight
"System 1" decision model: given a text and a question it returns
calibrated probabilities in a single forward pass, without generating
any text.

This is the single documentation home for the backend: parameter
guidance, measured results, and the tuning workflow.

## How it works

For each document the backend asks Laya **one choice question** —
*"Which subject does this text reference?"* — whose options are the
candidate subjects suggested by the configured source projects. Laya
compares the candidates against each other and returns a probability
for every option; each probability is blended with the candidate's
original source score to produce its final score.

```
sources ──> weighted merge (top-`limit` candidates)
              │
              ▼
       one Laya choice question per document
       (options = candidate labels)
              │
              ▼
       score = laya_weight · P(option)
             + (1 - laya_weight) · source score
              │
              ▼
       drop candidates below `threshold`, return
```

Fallbacks: if the prediction fails or the answer carries no
probabilities (for example when the options overflow the question token
budget), every candidate of that document keeps its original source
score. The models are never trained — the pipeline is fully zero-shot.

## Parameters

| parameter | meaning | default |
|---|---|---|
| `sources` | source projects, as in the plain ensemble | required |
| `limit` | option pool: how many merged candidates the choice question compares (and the cap on returned suggestions) | 20 |
| `laya_weight` | weight of Laya's option probability in the final score | 0.25 |
| `threshold` | minimum score of returned suggestions, applied inside the backend | 0.0 (YAKE sources: 0.75 — see "The YAKE profile") |
| `question_template` | the choice question's instruction text | "Which subject does this text reference?" |
| `laya_model` | checkpoint | `multilingual` (empty value restores language-based auto-routing) |
| `laya_max_len` / `laya_head_max_len` | token budgets (see below) | the checkpoint's own values |
| `laya_input_chars` | cap on the raw text sent to Laya (tokenizer savings) | 0 = full text |

### `limit` — the option pool

Keep it at or below ~20: Laya's own documentation warns that choice
accuracy degrades as the options outgrow the question token budget.
Keep it well above 2: measured on a 5,664-document corpus, a pool of
only the top-2 candidates caps gold recovery at 0.496, versus 0.799 at
depth 20 — the ceiling collapses before Laya even gets to rank. The
returned set size in practice is governed by `threshold`; callers can
cap it further per request (`-l` on the CLI, `limit` in the REST API),
as elsewhere in Annif.

### `laya_weight` — the blend

```
final score = laya_weight × P(option) + (1 − laya_weight) × source score
```

`laya_weight = 1.0` is pure replacement (the source ranking is
discarded); blending lets one weak signal be rescued by the other.
The 0.25 default is measured: it beat 0.5, 0.75 and 1.0 at every
candidate-pool depth on a weak source, and for already-strong sources
(where blending only hurts) it deviates least from the source ranking.

### `threshold` — output sizing

Applied to the final blended scores *inside* the backend, so every
caller — `annif eval`, `annif suggest`, the REST API — honors it; CLI
`-t` flags apply on top of it. Tune it with `annif optimize` (remove it
from the config first so the sweep sees the full score range), then set
it in the config before running `annif hyperopt`, which tunes the
weight at the configured threshold.

A threshold is only meaningful **paired with its weight**: the score
distribution depends on `laya_weight`. Never copy a threshold across
different weights. Thresholds can also sit on a cliff — measured
example: F1@5 jumps from 0.44 to 0.56 between threshold 0.74 and 0.75
at `laya_weight = 0.25` — so re-tune after changing the weight or
checkpoint.

For YAKE sources the measured optimum is **0.75 at `laya_weight =
0.25`** — start there and confirm with `annif optimize` (see "The
YAKE profile").

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
default. Doubling `max_len` doubles the GPU cost per question.

Neither budget is set by default: the checkpoint's own config applies —
with the default `multilingual` checkpoint that means 1024/256, reading
~3,000 characters. A productive override is running the multilingual
checkpoint at `512/192` (English reading depth): the multilingual model
is much lighter per token, measured at roughly 6x faster per document
at the same reading depth.

`laya_input_chars` complements the budgets: it truncates the raw text
*before* it is sent to Laya, so the tokenizer — which would otherwise
process the whole document on every call even though the model keeps
only its state budget of it — is not fed text that can never be read. A
value slightly above `(laya_max_len − laya_head_max_len) × 4`
characters per token is result-neutral for English documents, and must
be raised if the budgets are raised.

## Tuning workflow

Two Annif commands tune this backend, one axis each — chosen to match
the shape of each parameter's landscape:

| command | tunes | method |
|---|---|---|
| `annif optimize` | `threshold` (and per-call output sizing) | exhaustive grid — the right tool for the threshold's cliff-shaped landscape |
| `annif hyperopt` | `laya_weight`, at the configured limit and threshold | 1D adaptive search over the smooth weight axis; trials re-combine cached answers (~75 ms each) |

Step by step, on a validation corpus distinct from your test corpus —
values tuned on the test set overfit, which was observed repeatedly
during development:

1. **Tune the threshold** with `annif optimize`:

   ```bash
   annif optimize <project> <corpus> -m "F1 score (doc avg)"
   ```

   Apply its **threshold** recommendation to the project config (for
   YAKE sources it reliably lands at 0.75 — see "The YAKE profile"). Apply
   its **limit** recommendation *per call* (`-l N` on eval/suggest),
   not in the project config — the config's `limit` is the option pool,
   not an output cap. If a previous round set a `threshold`, remove it
   from the config first so the sweep sees the full score range.

2. **Tune the weight** with `annif hyperopt`, at that threshold:

   ```bash
   annif hyperopt <project> <corpus> -T 100 -m "F1 score (doc avg)"
   ```

   Any metric works for the weight (the threshold is fixed), including
   the default NDCG. Paste the single recommended `laya_weight` line
   into the config.

3. **Iterate if desired**: moving the weight shifts the score
   distribution, so re-run `annif optimize` (threshold removed) to
   re-tune the threshold at the new weight. One extra round usually
   suffices.

4. **Verify** with `annif eval`:

   ```bash
   annif eval <project> <corpus> -l <N> -M metrics.json
   ```

   Without `-l`, the CLI applies its default limit of 10. To measure
   the unthresholded pipeline, override the backend's threshold for the
   run: `-b laya.threshold=0.0`.

Use the same corpus and metric for both tuning commands.

## Measured results (5,664 documents, YAKE source)

The backend's value is proportional to the weakness of its source.

**Weak source (YAKE, raw P@1 0.41):** the blend lifts every ranking
metric at full scale:

| ranking | P@1 | P@3 | P@5 | F1@5 | NDCG |
|---|---|---|---|---|---|
| YAKE raw | 0.392 | 0.581 | 0.630 | 0.224 | 0.573 |
| YAKE + Laya, `laya_weight=0.25` | 0.546 | 0.745 | 0.768 | 0.266 | 0.685 |
| … with tuned `threshold=0.75` | — | — | — | **0.565** | — |

The threshold more than doubles F1@5 by shrinking returned sets to the
1–3 highest-confidence subjects. As a fully training-free pipeline
(YAKE extracts, Laya verifies), this is a strong zero-shot baseline
for languages or domains where no trained model exists.

**Strong source (MLLM, raw P@1 0.83):** the blend did *not* beat the
source (0.813 vs 0.828 P@1) in any variant tested — noul or choice
questions, blended or replaced, gated or not. Use a trained source when
one exists; use Laya where one does not.

Pool depth (measured, `laya_weight=0.25`): depth 10/15/20 gives
coverage 0.730/0.778/0.799 and F1@5 0.251/0.262/0.266; P@1 peaks at 15
(0.554). Depth beyond ~20 does not help: YAKE itself stops producing
candidates there, and Laya's option accuracy degrades.

Confidence gating (`min_confidence`) was implemented, measured, and
**removed**: the shipped checkpoints are over-confident, so the gate's
flagged answers were the helpful ones — gating at 0.75 made results
*worse* than no gate (P@1 0.808 gated vs 0.820 ungated vs 0.812
source).

## The YAKE profile

Both corpora tuned during development — independently, with different
tools — converged on the same operating point: **`laya_weight = 0.25`,
`threshold = 0.75`**. That is best understood not as a corpus-specific
tuning result but as the operating profile of the model pair
"YAKE + zero-shot Laya", and it has a mechanism:

- **The threshold follows YAKE's score distribution.** YAKE scores are
  document-relative, so the top keyword of almost any document lands
  near saliency 1.0. At `laya_weight = 0.25`, threshold 0.75 is exactly
  the boundary "saliency essentially perfect, unless Laya confirms" —
  a structural feature of YAKE's scoring, not of the text.
- **The weight follows the reliability ratio between the two models**:
  trust the zero-shot verifier about 25% against the extractor. That
  ratio is a property of the model pair, not of the corpus.

For a new YAKE-backed project, start at (0.25, 0.75), confirm the
threshold with one `annif optimize` run, and skip the weight search
unless the source is measurably stronger or weaker than usual — a
stronger source moves the optimum along a diagonal (lower weight,
higher threshold: on the stronger of the two corpora, (0.125, 0.87)
bought ~4pp of Precision@1 over the profile point).

**Why the defaults differ.** The backend defaults `laya_weight` to its
measured value (0.25) but `threshold` to the neutral 0.0. The weight's
optimum is a property of the model pair and held across both corpora,
so it is a safe default; the threshold's optimum depends on the
source's score scale and strength, and hardcoding 0.75 would silently
shrink the output of any project whose source is not YAKE-like. Set
the threshold per project: absent tuning, 0.0 is the no-surprise
default; for YAKE sources, 0.75 is the measured starting point.

## Operation notes

- **Not trainable.** Training the laya backend raises
  `NotSupportedException`; train the source projects instead.
- **Hyperopt internals.** `annif hyperopt` runs the source merge and
  the Laya questions once (`_prepare`), then re-combines the cached
  `(candidate, probability)` pairs for every trial — the trials cost no
  inference, so hundreds are cheap. The objective mirrors the backend's
  exact output path.
- **Progress logging.** One INFO line per batch:
  `Laya: 32 doc(s), 155 candidate(s), 1.30s`. Each failed prediction
  logs a warning; per-question details are at DEBUG level.
- **Performance.** Cost is one forward pass per document. On Apple
  Silicon the multilingual checkpoint at `512/192` measured ~40 ms per
  document; the English checkpoint at the same depth is ~6x slower.

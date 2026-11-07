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
*"What is the text about?"* — whose options are the candidate subjects
suggested by the configured source projects. Laya compares the
candidates against each other and returns a probability for every
option; each probability is blended with the candidate's original
source score to produce its final score.

```
sources ──> [optional, source_weights=true:
            one noul question per source,
            "Is the document about NAME?"
            → per-document source weights
              (blended, top-K gated)]
              │
              ▼
       weighted merge (top-`limit` candidates)
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
score. The pipeline is zero-shot as long as the project has not been
trained; `annif train` fine-tunes the checkpoint on a labeled corpus
(see "Training and fine-tuning").

## Parameters

Quick reference (as `--help` style; the table below has the details):

```
laya backend parameters
=======================
sources                   (required) source projects "id1:w1,id2:..." — static merge weights
limit                     20        choice-question option pool and output cap (subjects)
question_template         "What is the text about?"
                                    instruction of the per-document choice question
laya_weight               0.25      rerank blend: score = w·P(option) + (1−w)·source score
threshold                 0.0       minimum returned score, applied inside the backend
laya_model                "multilingual"
                                    checkpoint; empty = Laya routes by document language
laya_max_len              (unset)   token budget override for Router.predict
laya_head_max_len         (unset)   question-head token budget override
laya_input_chars          0         cap on raw text sent to Laya (0 = full text)
source_weights            false     true = per-document top-K source gating from Laya noul answers
source_question_template  "Is the document about {name}?"
                                    per-source noul instruction; {name}=project name (fallback id), {id}=id
source_top_k              0         keep only top-K sources per doc by noul score (0/≥n = keep all)

annif train only (forwarded to laya.train):
laya_epochs               1         fine-tuning epochs
laya_loss                 "soft-ce" fine-tuning objective (rlcd | soft-ce)
laya_micro_batch          16        fine-tuning micro batch size
laya_grad_accum           4         gradient accumulation steps (× micro_batch = effective batch)
laya_shuffle_options      false     shuffle choice options each epoch
laya_freeze_encoder       false     train only the decision head
laya_device               "auto"    Laya device for suggest-time inference and fine-tuning (auto|cpu|mps|cuda|...)
```

Notes: `limit` caps subjects, `source_top_k` caps sources per document;
with `source_weights=true` each document costs two Laya forward passes
(noul + choice; only the noul pass when `laya_weight=0`, pure MoE);
`annif hyperopt` tunes `laya_weight`; `annif optimize` tunes `threshold`.

| parameter | meaning | default |
|---|---|---|
| `sources` | source projects, as in the plain ensemble | required |
| `limit` | option pool: how many merged candidates the choice question compares (and the cap on returned suggestions) | 20 |
| `laya_weight` | weight of Laya's option probability in the final score | 0.25 |
| `threshold` | minimum score of returned suggestions, applied inside the backend | 0.0 (YAKE sources: 0.75 — see "The YAKE profile") |
| `question_template` | the choice question's instruction text | "What is the text about?" |
| `laya_model` | checkpoint | `multilingual` (empty value restores language-based auto-routing) |
| `laya_max_len` / `laya_head_max_len` | token budgets (see below); also cap fine-tuning, and Laya writes whatever it trained with into the checkpoint, so inference follows training | the checkpoint's own values |
| `laya_input_chars` | cap on the raw text sent to Laya (tokenizer savings) | 0 = full text |
| `source_weights` | boolean: `false` (default) = no source gating; `true` = per-document top-K gating from Laya noul answers (see below) | `false` |
| `source_question_template` | instruction of the per-source noul questions; `{name}` = source project's `name` (falls back to its id), `{id}` = project id | `"Is the document about {name}?"` |
| `source_top_k` | keep only the top-K sources per document ranked by noul P(true), with their static config weights (MoE-style gate); 0 or ≥ number of sources = keep all | 0 |
| `laya_epochs` | fine-tuning epochs (`annif train` only) | 1 |
| `laya_loss` | fine-tuning objective: `rlcd` or `soft-ce` | `soft-ce` |
| `laya_micro_batch` / `laya_grad_accum` | fine-tuning batch shape; their product is the effective batch (the notebook's is 64) | 16 / 4 |
| `laya_freeze_encoder` | train only the decision head, keep the encoder frozen (faster, less adaptable) | false |
| `laya_shuffle_options` | shuffle choice options each epoch during fine-tuning (breaks position priors) | false |
| `laya_device` | device for Laya inference at suggest time and for fine-tuning (`auto`, `cpu`, `mps`, `cuda`, …) | `auto` |

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

### Per-document source weighting (MoE-style gating)

With `source_weights=true` the merge weights are decided **per document**
instead of being fixed in `sources=`: for n sources the backend sends one
`noul` question per source — *"Is the document about NAME?"* — in a single
`predict()` call (one forward pass for all n questions). Each answer's
P(true) ranks that source; only the top `source_top_k` sources contribute
to the merge, with their static config weights (renormalized per document
over the survivors, mirroring how `parse_sources` normalizes the static
weights). NAME is the source project's `name=` from its config, falling
back to the project id. The questions only make sense when the names denote
topics or domains, so name source projects accordingly.

A gated-out source contributes 0 to the merge, so its unique candidates
drop out of that document's option pool for the choice question. This
reduces false positives by shrinking the pool before the choice question.

Top-K is preferred over an absolute threshold gate because it ranks sources
against each other per document and is therefore independent of the
checkpoint's probability calibration (the shipped checkpoints are
over-confident — see "Measured results"). `source_top_k=1` is pure
winner-take-all routing per document. With disjoint-domain sources of
equal reliability (e.g. one source per occupation domain, all with the
default equal weights), the survivors' equal config weights renormalize
over the kept experts — Laya routes, the config weighs.

Fallbacks, mirroring the choice pass: if the noul `predict()` fails for a
document, that document keeps the static config weights (and the top-K gate
does not apply to the fallback); a missing answer for one source gives that
source its config weight; if the weights would sum to zero, the config
weights are used. A single source skips the noul pass entirely. When the
mode is on, the cost is **two forward passes per document** (noul pass +
choice pass) — they cannot be fused into one call because the choice
options depend on the weighted merge, which depends on the noul answers.

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

## Training and fine-tuning

The backend is zero-shot until you train it. `annif train` fine-tunes
the Laya checkpoint itself, using Laya's own training recipe
(`laya.train.finetune`) on the project corpus:

```bash
annif train <project> <corpus>
```

Training data is built the same way the backend works at suggest time:
for every document the source projects are asked for suggestions, the
merged candidates become the options of the same choice question the
backend asks at suggest time, and the document's gold subjects that
appear among the options become the answer. Documents whose gold
subjects include none of the candidates are skipped — Laya cannot learn
from a question whose answer is not among the options — and training
fails with `NotSupportedException` when no document yields a row.

With `source_weights=true` (and more than one source), each training row
additionally carries the per-source **noul questions**, so the
fine-tuned checkpoint learns the routing it will be asked to do at
suggest time. Their gold is derived, not asked: P(true)=1 for every
source whose own suggestions include a gold subject of the document,
0 otherwise. Two deliberate deviations from suggest time:

- The choice-option pool is built from the **static** merge, not the
  per-document weighted one. A stock-checkpoint-gated pool would drop
  exactly the documents the stock routing gets wrong (their gold
  candidates fall outside the pool), biasing training toward documents
  the router already handles; the static pool keeps gold coverage
  maximal. As a side effect, training-data generation performs **no
  Laya inference at all** — the noul gold comes from the source
  batches, not from Laya answers.
- The noul gold sets both `"false"` and `"true"` explicitly, because
  Laya's trainer defaults a missing noul key to 0.5.

The fine-tuned checkpoint is saved under the project datadir
(`laya-model/`) and used **automatically** afterwards, as long as the
project config does not set `laya_model` explicitly (an explicit
`laya_model` always wins, so a trained project can still be pointed at
a stock checkpoint). `is_trained` reports a fine-tuned checkpoint as
trained; without one it falls back to the trained state of the source
projects, which is what the zero-shot pipeline needs.

Notes:

- The base checkpoint for fine-tuning is the configured `laya_model`
  (default `multilingual`); with an empty `laya_model` (language-based
  routing) fine-tuning starts from `multilingual`, since it needs one
  concrete checkpoint.
- `--cached` (`annif train -c`) reuses the preprocessed training rows
  from the previous run (`laya-train.jsonl` in the project datadir)
  and only re-runs the fine-tuning itself.
- Training-data generation parallelizes across document batches with
  `annif train -j N` (0 = all CPUs); only the `finetune` call itself runs
  on one device (`laya_device`, default `auto`). The workers need no
  Laya model: the merge is static and the noul gold is derived from the
  source batches.
- At suggest/eval time with `-j N`, the router is loaded lazily on first
  rescore in each worker process; the main process skips loading it
  entirely when initialized with `parallel=True`. Set `laya_device`
  (e.g. `mps` on Apple Silicon) to choose the inference device.
- **MPS workaround.** Upstream laya (0.3.29, 0.4.18) runs its pre-training
  calibration forward before moving the model to the target device, so
  on Apple Silicon with multi-question training rows (i.e.
  `source_weights=true`) it crashes with `Passed CPU tensor to MPS op`.
  The backend monkeypatches `laya.train._forward` to move the model to
  the batch device first (a no-op when it already is); remove the patch
  in `_patch_laya_train_forward_device` once upstream fixes it. Until
  then, `laya_device=cpu` also works around it.
- The default epoch budget (`laya_epochs = 4`) is sized for a few
  thousand labeled documents; on a few hundred rows the Laya trainer
  itself warns when predictions collapse to the class prior — use more
  epochs, more data, or another seed.
- `laya_shuffle_options` re-encodes choice questions with a random
  option order each epoch, breaking the position prior that fixed-order
  options teach; consider it for large corpora.
- After fine-tuning, re-run the tuning workflow below: the score
  distribution of a fine-tuned checkpoint differs from the stock one,
  so `threshold` and `laya_weight` need re-measuring. All Laya questions
  — noul and choice — go to the fine-tuned checkpoint afterwards.

## Tuning workflow

Two Annif commands tune this backend, one axis each — chosen to match
the shape of each parameter's landscape:

| command | tunes | method |
|---|---|---|
| `annif optimize` | `threshold` (and per-call output sizing) | exhaustive grid — the right tool for the threshold's cliff-shaped landscape |
| `annif hyperopt` | `laya_weight`, at the configured limit and threshold | 1-D adaptive search over the smooth weight axis; trials re-combine cached answers (~75 ms each) |

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

### Choosing what to tune: reranking, MoE, or both

The knobs split by which pipeline stage they act on, and that determines
what you tune them with:

| knob | stage | default | tuned by |
|---|---|---|---|
| `source_top_k` | MoE gate (sources, per document) | 0 = off | manual (small integer space) — not auto-tuned |
| `limit` | candidate pool (subjects) | 20 | per-call `-l`; `optimize` recommends |
| `laya_weight` | rerank blend (subjects) | 0.25 | `annif hyperopt` (first axis) |
| `threshold` | output filter on the final score | 0.0 | `annif optimize` grid |

**Reranking only** (`source_weights=false`, the default): the existing
workflow above — `optimize` for `threshold`, `hyperopt` 1-D for
`laya_weight`, iterate once. The measured (0.25, 0.75) YAKE profile
applies here unchanged.

**MoE gate + reranking (both on)**: `hyperopt` stays 1-D over
`laya_weight` (the gate is fixed during optimization).
`source_top_k` stays manual: it is a discrete, per-document cliff knob
that also reshapes the option pool, so it is the worst fit for a smooth
sampler. With n sources its space is just 1..n−1, so sweep it by hand
(K=1 is pure winner-take-all routing). An aggressive K also shrinks the
pool below `limit` (`limit` is a cap, not a target), which can lower gold
coverage — the measured coverage ceiling at pool depth 20 only holds if
the gate does not starve the merge.

**MoE only**: set `laya_weight=0.0` — the choice pass is then skipped
entirely (no reranking inference is paid), and the output is the
weighted-merged scores filtered by `limit`/`threshold`.

Cross-cutting rules that span both modes:

- **`threshold` must be re-tuned after changing any weight** — it filters
  the final blended score, whose distribution depends on `laya_weight`
  *and* on the MoE weights. Remove it from the config before
  `annif optimize` so the grid sees the full score range; it is
  cliff-shaped, so the grid (not hyperopt) is the right tool.
- **Cost of tuning**: with MoE on, `hyperopt`'s `_prepare` pays both Laya
  passes once per corpus and trials stay inference-free; but
  `annif optimize`'s threshold grid re-runs suggest per grid point, so
  each grid point costs two forward passes per document.

Practical recipe for both modes: fix `source_top_k` by hand →
`annif hyperopt` (2-D weights) → `annif optimize` (threshold) → optionally
re-run hyperopt at the new threshold.

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

- **Trainable by fine-tuning.** `annif train` fine-tunes the Laya
  checkpoint on a labeled corpus (see "Training and fine-tuning");
  train the source projects separately as usual. Without training the
  pipeline is zero-shot.
- **Hyperopt internals.** `annif hyperopt` runs the source merge and
  the Laya questions once (`_prepare`), then re-combines the cached
  `(candidate, probability)` pairs for every trial — the trials cost no
  inference, so hundreds are cheap. The objective mirrors the backend's
  exact output path, and the questions go to the fine-tuned checkpoint
  when one has been trained.
- **Progress logging.** One INFO line per batch:
  `Laya: 32 doc(s), 155 candidate(s), 1.30s`, and with
  `source_weights=true` also `Laya source weights: 32 doc(s), 2
  source(s), 1.28s`. Each failed prediction logs a warning; per-question
  details are at DEBUG level.
- **Performance.** Cost is one forward pass per document, or two when
  `source_weights=true` is enabled (the noul pass cannot be fused with
  the choice pass — see above). On Apple
  Silicon the multilingual checkpoint at `512/192` measured ~40 ms per
  document; the English checkpoint at the same depth is ~6x slower.

"""Ensemble backend that re-scores source suggestions with the Laya
decision model (https://laya.convaiinnovations.com, Hugging Face:
convaiinnovations/laya, PyPI package ``laya``).

Like the plain ensemble backend it merges the suggestions of the
configured source projects, but instead of returning the merged scores
as-is, it asks Laya one "choice" question per document ("What is the
text about?") with the candidate subjects as options, and blends each
option probability with the candidate's source score. The pipeline is
zero-shot by default; ``annif train`` can fine-tune the checkpoint on a
labeled corpus with Laya's own training recipe (``laya.train``), after
which the fine-tuned checkpoint is used automatically.

With ``source_weights=true`` the merge is gated per document
(MoE-style): Laya is asked one "noul" question per source ("Is the
document about NAME?"), the answers rank the sources, and only the top
``source_top_k`` sources (by P(true)) contribute to the merge, with
their static config weights - before the choice-question re-scoring
runs on the re-merged candidates. ``annif train`` fine-tunes the noul
questions too in that mode (gold derived from which sources suggested
gold subjects), over a statically merged candidate pool.

Parameter documentation, measured results and the tuning workflow
(``annif optimize`` / ``annif hyperopt``) live in docs/laya.md.
Requires the ``laya`` Python package; the checkpoint is downloaded
from Hugging Face on first use.

YAKE + Laya works with the following languages:

Arabic	ar
Danish	da
German	de
Greek	el
English	en
Spanish	es
Persian	fa
Finnish	fi
French	fr
Hindi	hi
Hungarian	hu
Armenian	hy
Indonesian	id
Italian	it
Japanese	ja
Latvian	lv
Dutch	nl
Polish	pl
Portuguese	pt
Romanian	ro
Russian	ru
Slovenian	sl
Swedish	sv
Turkish	tr
Chinese	zh
"""

from __future__ import annotations

import copy
import json
import os.path
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

import annif.eval
import annif.parallel
import annif.util
from annif.exception import (
    NotInitializedException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch, SuggestionResult

from . import hyperopt
from .ensemble import BaseEnsembleBackend

if TYPE_CHECKING:
    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.corpus.document import Document, DocumentCorpus
    from annif.vocab.types import Subject


class LayaHPObjective(hyperopt.HPObjective):
    """Objective of the laya hyperparameter optimizer: samples
    laya_weight values and re-combines the Laya answers cached by
    _prepare() exactly like the backend does at suggest time - at the
    configured limit and threshold - so trials need no model
    inference."""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        laya_weight = trial.suggest_float("laya_weight", 0.0, 1.0)
        limit = args["limit"]
        threshold = args["threshold"]

        eval_batch = annif.eval.EvaluationBatch(args["subject_index"])
        for verified, gold_batch in zip(args["verified_batches"], args["gold_batches"]):
            rescored_results = []
            for pairs, _status in verified:
                suggestions = []
                for suggestion, probability in pairs:
                    if probability is None:
                        score = suggestion.score
                    else:
                        score = (
                            laya_weight * probability
                            + (1.0 - laya_weight) * suggestion.score
                        )
                    suggestions.append(
                        SubjectSuggestion(subject_id=suggestion.subject_id, score=score)
                    )
                rescored_results.append(suggestions)
            batch = SuggestionBatch.from_sequence(
                rescored_results, args["subject_index"]
            )
            batch = batch.filter(limit=limit, threshold=threshold)
            eval_batch.evaluate_many(batch, gold_batch)

        return eval_batch.results(metrics=[args["metric"]])[args["metric"]]


class LayaOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the laya backend: selects
    laya_weight for a labeled corpus, at the limit and threshold
    configured in the project. _prepare() runs the source merge and the
    Laya questions once; the trials re-combine the cached answers, so
    any number of trials costs one inference pass over the corpus.
    Tune the threshold with "annif optimize" instead (its grid handles
    the threshold's cliff-shaped landscape); after this optimizer moves
    the weight, re-run optimize to re-tune the threshold at the new
    score distribution."""

    def __init__(
        self,
        backend: LayaBackend,
        corpus: DocumentCorpus,
        metric: str,
    ) -> None:
        super().__init__(backend, corpus, metric, LayaHPObjective)

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        # n_jobs is unused: preparation is bound by the single Laya
        # forward pass per document, not by CPU parallelism
        self._backend.initialize()
        params = self._backend._get_backend_params(None)
        sources = annif.util.parse_sources(params["sources"])

        verified_batches = []
        gold_batches = []
        for doc_batch in self._corpus.doc_batches:
            docs = list(doc_batch)
            batch_by_source = self._backend._suggest_with_sources(docs, sources)
            merged = self._backend._merge_for_documents(
                docs, batch_by_source, sources, params
            )
            verified_batches.append(self._backend._verify_batch(docs, merged, params))
            gold_batches.append([doc.subject_set for doc in docs])

        return {
            "verified_batches": verified_batches,
            "gold_batches": gold_batches,
            "subject_index": self._backend.project.subjects,
            "limit": int(params["limit"]),
            "threshold": float(params.get("threshold") or 0.0),
            "metric": self._metric,
        }

    def _postprocess(self, study: Study) -> hyperopt.HPRecommendation:
        return hyperopt.HPRecommendation(
            lines=["laya_weight = {:.4f}".format(study.best_params["laya_weight"])],
            score=study.best_value,
        )


class TrainDataWorker(annif.parallel.BaseWorker):
    """Builds fine-tuning rows for document batches in parallel workers.

    args = (backend, sources, params, use_weights). Each worker
    initializes only the source projects — no Laya model is loaded,
    because training-data generation uses the static merge and derives
    the noul gold from the source batches, never from Laya."""

    @classmethod
    def init(cls, args) -> None:
        super().init(args)
        backend, sources, _params, _use = args
        for source_id, _ in sources:
            backend.project.registry.get_project(source_id).initialize()

    @classmethod
    def process_batch(cls, doc_batch):
        backend, sources, params, use_weights = cls.args
        docs = list(doc_batch)
        batch_by_source = backend._suggest_with_sources(docs, sources)
        # static pool on purpose: maximal gold coverage, no stock-routing
        # bias (see _write_train_data)
        merged = backend._merge_source_batches(batch_by_source, sources, params)
        noul_questions = None
        aligned_by_source = None
        if use_weights:
            noul_questions = backend._source_noul_questions(sources, params)
            aligned_by_source = {
                source_id: backend._align_batch(batch_by_source[source_id], source_id)
                for source_id, _ in sources
            }
        rows = []
        skipped = 0
        for doc_index, (doc, result) in enumerate(zip(docs, merged)):
            row = backend._train_row(
                doc,
                doc_index,
                result,
                sources,
                params,
                noul_questions,
                aligned_by_source,
            )
            if row is None:
                skipped += 1
            else:
                rows.append(json.dumps(row, ensure_ascii=False))
        return rows, skipped


class LayaBackend(BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that re-scores merged suggestions with Laya by
    asking one choice question whose options are the suggested
    subjects. With source_weights=true the source merge itself is
    weighted per document by Laya noul answers (see _merge_for_documents).
    Without training the pipeline is zero-shot; annif train fine-tunes
    the checkpoint on a labeled corpus."""

    name = "laya"

    # fine-tuned checkpoint directory (inside the project datadir) and
    # training data file, both written by _train()
    MODEL_DIR = "laya-model"
    TRAIN_FILE = "laya-train.jsonl"
    # name the fine-tuned checkpoint is registered under on the Router
    FINETUNED_MODEL = "finetuned"

    # See docs/laya.md for parameter documentation and measurements.
    DEFAULT_PARAMETERS = {
        # option pool for the choice question (and cap on suggestions)
        "limit": 20,
        # instruction of the choice question
        "question_template": "What is the text about?",
        # blend weight: score = w * option_probability + (1 - w) * source
        "laya_weight": 0.25,
        # minimum returned score, applied inside the backend
        "threshold": 0.0,
        # checkpoint; set to an empty value to let Laya route by language
        "laya_model": "multilingual",
        # cap on the raw text sent to Laya (tokenizer savings)
        "laya_input_chars": 0,
        # per-document source gating (MoE-style): when true, ask one noul
        # question per source and keep only the top source_top_k sources
        # per document (ranked by the noul answer); survivors keep their
        # static config weights; when false (default), no gating
        "source_weights": False,
        # instruction of the per-source noul questions; {name} is the
        # source project's name (falling back to its id), {id} the id
        "source_question_template": "Is the document about {name}?",
        # keep only the top-K sources per document by noul score
        # (0 or >= the number of sources: keep all)
        "source_top_k": 0,
        # fine-tuning knobs, forwarded to laya.train (annif train
        # only); defaults are tuned for speed on large corpora, not
        # for maximum quality (see docs/laya.md)
        "laya_epochs": 1,
        "laya_loss": "soft-ce",
        "laya_micro_batch": 16,
        "laya_grad_accum": 4,
        "laya_shuffle_options": False,
        "laya_freeze_encoder": False,
        "laya_device": "auto",
    }

    # class-level default so uninitialized instances behave like other
    # backends (pattern from MLLMBackend._model)
    _router = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        if parallel:
            # defer loading the Laya model: with a process pool, rescoring
            # happens in the workers, so the main process never needs it
            return
        self._ensure_router(self.params)

    def _ensure_router(self, params: dict[str, Any]) -> None:
        """Load the Laya decision model on first use, so that parallel
        initialization (and the main process of a pooled run) does not pay
        for the checkpoint unless rescoring actually happens here."""
        if self._router is not None:
            return
        self.info("loading Laya decision model (downloads checkpoint on first use)")
        try:
            # imported lazily so Annif runs without the package
            from laya import Router
        except ImportError as err:
            raise OperationFailedException(
                "Laya package not available, cannot use laya backend"
            ) from err
        device = params.get("laya_device", self.DEFAULT_PARAMETERS["laya_device"])
        self._router = Router(device=None if device == "auto" else device)
        self.info("Laya decision model loaded")
        finetuned = self._finetuned_model_dir
        if finetuned is not None:
            self._router.register(
                self.FINETUNED_MODEL,
                finetuned,
                description="checkpoint fine-tuned by annif train",
            )
            self.info("using fine-tuned Laya checkpoint {}".format(finetuned))

    def _subject_label(self, subject: Subject) -> str:
        """Best available label for a subject: the vocabulary language
        first, then any label, then the URI."""
        labels = subject.labels or {}
        label = labels.get(self.project.vocab_lang)
        if not label and labels:
            label = next(iter(labels.values()))
        return label or subject.uri

    @property
    def _finetuned_model_dir(self) -> str | None:
        """Path to the fine-tuned checkpoint written by annif train, or
        None when the backend has not been trained."""
        path = os.path.join(self.datadir, self.MODEL_DIR)
        if os.path.exists(os.path.join(path, "rl_agent_config.json")):
            return path
        return None

    def _resolve_model(self, params: dict[str, Any]) -> str | None:
        """The checkpoint name to ask the Router for: the fine-tuned
        model when one has been trained and the config does not pin
        laya_model, otherwise the configured laya_model."""
        if "laya_model" not in self.config_params and self._finetuned_model_dir:
            return self.FINETUNED_MODEL
        return params.get("laya_model")

    def _predict_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        """Optional Router.predict() keyword arguments from the backend
        parameters: checkpoint (laya_model, or the fine-tuned model when
        one exists) and token budgets (laya_max_len, laya_head_max_len)."""
        kwargs = {}
        model = self._resolve_model(params)
        if model:
            kwargs["model"] = model
        if params.get("laya_max_len"):
            kwargs["max_len"] = int(params["laya_max_len"])
        if params.get("laya_head_max_len"):
            kwargs["head_max_len"] = int(params["laya_head_max_len"])
        return kwargs

    def _source_noul_questions(
        self, sources: list[tuple[str, float]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """One noul question per source, keyed by project id. The
        answer's P(true) becomes the source's weight for the document.
        {name} in the template is the source project's name (Project.name
        falls back to the project id), {id} the project id."""

        template = params.get(
            "source_question_template",
            self.DEFAULT_PARAMETERS["source_question_template"],
        )
        questions = {}
        for project_id, _ in sources:
            project = self.project.registry.get_project(project_id)
            instructions = template.replace("{name}", project.name).replace(
                "{id}", project_id
            )
            questions[project_id] = {"type": "noul", "instructions": instructions}
        return questions

    def _source_noul_probabilities(
        self,
        documents: list[Document],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> tuple[list[list[float | None]], list[str]]:
        """Ask Laya the per-source noul questions: one predict call per
        document carrying all n source questions, so the whole pass is
        a single forward pass per document.

        Returns per document a list of P(true) per source (None when the
        answer lacks that question) and a status: "ok", "failed" (the
        predict call raised; the document falls back to config weights)
        or "no_probabilities"."""

        questions = self._source_noul_questions(sources, params)
        input_chars = int(params.get("laya_input_chars", 0))
        predict_kwargs = self._predict_kwargs(params)
        started = time.monotonic()

        all_probs = []
        statuses = []
        for doc in documents:
            # laya_input_chars truncates the text sent to Laya only, as
            # in the choice pass; the sources always saw the full text
            text = doc.text[:input_chars] if input_chars > 0 else doc.text
            try:
                answer = self._router.predict(text, questions, **predict_kwargs)
                answers = answer["answers"]
            except Exception as err:
                # keep the static config weights for this document rather
                # than failing the whole suggestion batch
                self.warning(
                    "Laya source weighting failed, keeping config weights: "
                    "{}".format(err)
                )
                all_probs.append([None] * len(sources))
                statuses.append("failed")
                continue
            row = []
            missing = False
            for project_id, _ in sources:
                value = answers.get(project_id)
                prob = value.get("noul") if isinstance(value, dict) else None
                if prob is None:
                    missing = True
                row.append(float(prob) if prob is not None else None)
            all_probs.append(row)
            statuses.append("no_probabilities" if missing else "ok")

        self.info(
            "Laya source weights: {} doc(s), {} source(s), {:.2f}s".format(
                len(documents), len(sources), time.monotonic() - started
            )
        )
        return all_probs, statuses

    def _gate_source_weights(
        self,
        noul_row: list[float | None],
        config_weights: list[float],
        params: dict[str, Any],
    ) -> np.ndarray:
        """One document's source weights for the top-K gate: rank sources
        by their noul score P(true) (a missing answer ranks last), keep
        the source_top_k highest, and give survivors their static config
        weight — Laya routes, the config weighs.

        A document whose noul pass failed (all None) and a document whose
        weights would sum to zero fall back to the config weights. The
        returned weights are not normalized; _merge_with_weights
        normalizes per row."""

        top_k = int(params.get("source_top_k", self.DEFAULT_PARAMETERS["source_top_k"]))
        cw = np.asarray(config_weights, dtype=np.float64)
        if all(p is None for p in noul_row):
            return cw.copy()
        scores = np.array([p if p is not None else -1.0 for p in noul_row])
        n = len(cw)
        weights = cw.copy()
        if 0 < top_k < n:
            # deterministic tie-break: noul score desc, config weight desc,
            # then source order
            order = sorted(range(n), key=lambda i: (-scores[i], -cw[i], i))
            keep = set(order[:top_k])
            weights = np.array([cw[i] if i in keep else 0.0 for i in range(n)])
        if weights.sum() <= 0.0:
            return cw.copy()
        return weights

    def _merge_with_weights(
        self,
        batches: list[SuggestionBatch],
        weights: np.ndarray,
        limit: int,
    ) -> SuggestionBatch:
        """Weighted average of aligned source batches with a
        per-document weight matrix (n_documents x n_sources), normalized
        per row like SuggestionBatch.from_averaged normalizes its scalar
        weights."""

        from scipy.sparse import diags

        from annif.suggestion import csr_array

        weights = np.asarray(weights, dtype=np.float64)
        rowsum = weights.sum(axis=1)
        rowsum[rowsum <= 0.0] = 1.0
        total = None
        for i, batch in enumerate(batches):
            scaled = csr_array(diags(weights[:, i]) @ batch.array)
            total = scaled if total is None else total + scaled
        merged = csr_array(diags(1.0 / rowsum) @ total)
        # a gated-out source leaves explicit zeros behind; drop them so
        # its candidates cannot re-enter the option pool with score 0
        merged.eliminate_zeros()
        return SuggestionBatch(merged.astype(np.float32)).filter(limit=limit)

    def _weighted_source_merge(
        self,
        documents: list[Document],
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> tuple[SuggestionBatch, list[SuggestionBatch], list[list[float | None]]]:
        """Merge the source batches with the per-document top-K gate: ask
        the noul source questions, rank the sources by the answers, keep
        the top source_top_k with their config weights, and merge with
        the resulting weight matrix.

        Returns (merged batch, aligned source batches, raw noul
        probabilities)."""

        config_weights = [weight for _, weight in sources]
        noul, _statuses = self._source_noul_probabilities(documents, sources, params)
        weights = np.array(
            [self._gate_source_weights(row, config_weights, params) for row in noul]
        )
        top_k = int(params.get("source_top_k", self.DEFAULT_PARAMETERS["source_top_k"]))
        if 0 < top_k < len(sources):
            for doc_idx, row in enumerate(noul):
                if all(p is None for p in row):
                    continue
                dropped = [
                    project_id
                    for (project_id, _), w in zip(sources, weights[doc_idx])
                    if w == 0.0
                ]
                if dropped:
                    self.debug(
                        "doc {}: top-{} gate dropped source(s) {}".format(
                            doc_idx, top_k, ",".join(dropped)
                        )
                    )
        batches = [
            self._align_batch(batch_by_source[project_id], project_id)
            for project_id, _ in sources
        ]
        merged = self._merge_with_weights(batches, weights, int(params["limit"]))
        return merged, batches, noul

    def _use_source_weights(
        self, sources: list[tuple[str, float]], params: dict[str, Any]
    ) -> bool:
        """Whether per-document Laya source weighting applies: enabled
        with source_weights=true and meaningful only with more than one
        source (a single source's weight always normalizes to 1)."""

        return (
            annif.util.boolean(params.get("source_weights", False))
            and len(sources) >= 2
        )

    def _merge_for_documents(
        self,
        documents: list[Document],
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        """Merge the source batches, using per-document Laya weights when
        source_weights=true and there is more than one source to weigh;
        otherwise the plain static merge of the ensemble backend."""

        if not self._use_source_weights(sources, params):
            return self._merge_source_batches(batch_by_source, sources, params)
        self._ensure_router(params)
        merged, _, _ = self._weighted_source_merge(
            documents, batch_by_source, sources, params
        )
        return merged

    def _choice_question(
        self, suggestions: list[SubjectSuggestion], params: dict[str, Any]
    ) -> tuple[dict[str, SubjectSuggestion], dict[str, Any]]:
        """Build the choice question over the candidate suggestions.

        Returns (option_text, questions): option_text maps the unique
        option label of each suggestion to the suggestion itself, and
        questions is the question dict Router.predict() expects. Option
        texts must be unique: duplicate labels get a counter suffix,
        because Laya answers are keyed by these texts."""

        default_template = self.DEFAULT_PARAMETERS["question_template"]
        instructions = params.get("question_template", default_template)

        option_text = {}
        for suggestion in suggestions:
            subject = self.project.subjects[suggestion.subject_id]
            base = self._subject_label(subject)
            label, n = base, 1
            while label in option_text:
                n += 1
                label = "{} ({})".format(base, n)
            option_text[label] = suggestion

        questions = {
            "the_subject": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {label: "" for label in option_text},
            }
        }
        return option_text, questions

    def _verify_batch(
        self,
        documents: list[Document],
        merged: SuggestionBatch,
        params: dict[str, Any],
    ) -> list[tuple[list[tuple[SubjectSuggestion, float | None]], str]]:
        """Ask Laya to verify the merged candidates: one choice question
        per document, options = candidate labels.

        Returns one (pairs, status) entry per document, where pairs are
        (suggestion, probability_or_None) in source-ranking order and
        status is "ok", "failed", "no_probabilities" or "empty".
        Candidates with probability None keep their source score. This
        is the expensive half of the backend (one Laya forward pass per
        document); the hyperparameter optimizer reuses it to cache the
        answers once for all its trials."""

        self._ensure_router(params)
        input_chars = int(params.get("laya_input_chars", 0))
        predict_kwargs = self._predict_kwargs(params)

        verified = []
        for doc, result in zip(documents, merged):
            suggestions = list(result)

            if not suggestions:
                verified.append(([], "empty"))
                continue

            option_text, questions = self._choice_question(suggestions, params)

            # laya_input_chars: the sources always see the full text,
            # only the question may use the truncated text
            text = doc.text[:input_chars] if input_chars > 0 else doc.text

            self.debug(
                "asking Laya a choice question with {} option(s) for a "
                "document of {} characters".format(len(option_text), len(doc.text))
            )

            try:
                answer = self._router.predict(text, questions, **predict_kwargs)
                value = answer["answers"]["the_subject"]
            except Exception as err:
                # keep the plain ensemble scores rather than failing the
                # whole suggestion batch
                self.warning(
                    "Laya prediction failed, keeping original scores: {}".format(err)
                )
                verified.append(([(s, None) for s in suggestions], "failed"))
                continue

            probabilities = value.get("probabilities")
            if not probabilities:
                self.debug(
                    "  no probabilities in answer, keeping {} original "
                    "score(s)".format(len(suggestions))
                )
                verified.append(([(s, None) for s in suggestions], "no_probabilities"))
                continue

            pairs = []
            for label, suggestion in option_text.items():
                probability = probabilities.get(label)
                pairs.append(
                    (
                        suggestion,
                        float(probability) if probability is not None else None,
                    )
                )
            verified.append((pairs, "ok"))

        return verified

    def _rescore_with_laya(
        self,
        documents: list[Document],
        merged: SuggestionBatch,
        params: dict[str, Any],
    ) -> SuggestionBatch:
        """Blend the Laya option probabilities (from _verify_batch) with
        the source scores, then apply the output limit and threshold.
        With laya_weight=0.0 the choice pass is skipped entirely (pure
        MoE mode: the output is the weighted merge, no reranking
        inference is paid)."""

        laya_weight = float(
            params.get("laya_weight", self.DEFAULT_PARAMETERS["laya_weight"])
        )
        threshold = float(params.get("threshold") or 0.0)

        if laya_weight == 0.0:
            # no reranking: keep the (weighted) merged scores as-is
            return merged.filter(limit=int(params["limit"]), threshold=threshold)

        started = time.monotonic()

        verified = self._verify_batch(documents, merged, params)

        rescored_results = []
        candidates = 0
        for pairs, status in verified:
            if status == "empty":
                rescored_results.append([])
                continue

            candidates += len(pairs)

            if status in ("failed", "no_probabilities"):
                # no usable answer: keep the source scores
                rescored_results.append([suggestion for suggestion, _ in pairs])
                continue

            new_suggestions = []
            for suggestion, probability in pairs:
                if probability is None:
                    # option missing from the answer: keep the source score
                    new_suggestions.append(suggestion)
                else:
                    score = (
                        laya_weight * probability
                        + (1.0 - laya_weight) * suggestion.score
                    )
                    new_suggestions.append(
                        SubjectSuggestion(subject_id=suggestion.subject_id, score=score)
                    )
            rescored_results.append(new_suggestions)

        self.info(
            "Laya: {} doc(s), {} candidate(s), {:.2f}s".format(
                len(documents), candidates, time.monotonic() - started
            )
        )

        # filter after building the rows: the output must keep the
        # top-limit candidates by blended score, not by the source
        # ranking they arrived in
        batch = SuggestionBatch.from_sequence(rescored_results, self.project.subjects)
        return batch.filter(limit=int(params["limit"]), threshold=threshold)

    def _suggest_batch(
        self, documents: list[Document], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        batch_by_source = self._suggest_with_sources(documents, sources)
        merged = self._merge_for_documents(documents, batch_by_source, sources, params)
        return self._rescore_with_laya(documents, merged, params)

    @property
    def is_trained(self) -> bool:
        # trained when a fine-tuned checkpoint exists (annif train) or,
        # in the zero-shot pipeline, when all source projects are trained
        if self._finetuned_model_dir is not None:
            return True
        return all(self._get_sources_attribute("is_trained"))

    @property
    def modification_time(self) -> datetime | None:
        finetuned = self._finetuned_model_dir
        if finetuned is not None:
            config_file = os.path.join(finetuned, "rl_agent_config.json")
            return datetime.fromtimestamp(
                os.path.getmtime(config_file), tz=timezone.utc
            )
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def _write_train_data(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 1,
    ) -> None:
        """Write the Laya fine-tuning data for the corpus: a JSONL file
        of {state, questions, gold} rows, one row per document whose
        gold subjects include at least one of the merged candidates.

        The choice question of each row is the same one the backend asks
        at suggest time, but over the STATIC merge: the training pool
        must not be shaped by the stock checkpoint's noul answers, which
        would drop exactly the documents the stock routing gets wrong.
        When per-document source weighting is enabled, each row also
        carries the per-source noul questions with derived gold, so the
        fine-tuned checkpoint learns the routing it will be asked to do
        (see _train_row).

        Rows are built per document batch in parallel workers (jobs; 0 =
        all CPUs) via TrainDataWorker. The workers need no Laya model:
        the merge is static and the noul gold comes from the source
        batches, so only the source projects are initialized there."""

        sources = annif.util.parse_sources(params["sources"])
        for project_id, _ in sources:
            self.project.registry.get_project(project_id).initialize()

        use_weights = self._use_source_weights(sources, params)
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        # workers never call Laya: hand them a shallow copy without the
        # router, so process pools never pickle a loaded model, and the
        # main-process backend keeps its router when jobs=1 (threads)
        worker_backend = copy.copy(self)
        worker_backend._router = None
        worker_args = (worker_backend, sources, params, use_weights)

        rows = 0
        skipped = 0
        n_jobs, pool_class = annif.parallel.get_pool(jobs)
        with open(train_path, "w", encoding="utf-8") as train_file:
            with pool_class(
                n_jobs, initializer=TrainDataWorker.init, initargs=(worker_args,)
            ) as pool:
                for batch_rows, batch_skipped in pool.imap_unordered(
                    TrainDataWorker.process_batch, corpus.doc_batches
                ):
                    for line in batch_rows:
                        print(line, file=train_file)
                    rows += len(batch_rows)
                    # Laya cannot learn from a question whose answer
                    # is not among the options
                    skipped += batch_skipped
        self.info(
            "wrote {} Laya fine-tuning row(s) to {}, skipped {} document(s) "
            "without a gold candidate".format(rows, train_path, skipped)
        )
        if rows == 0:
            raise NotSupportedException(
                "no document had a gold subject among the merged candidates"
            )

    def _train_row(
        self,
        doc: Document,
        doc_index: int,
        result: SuggestionResult,
        sources: list[tuple[str, float]],
        params: dict[str, Any],
        noul_questions: dict[str, Any] | None,
        aligned_by_source: dict[str, SuggestionBatch] | None,
    ) -> dict[str, Any] | None:
        """Build one fine-tuning row ({state, questions, gold}) for a
        document, or None when no gold subject is among the candidates.

        The row carries the choice question over the merged candidates
        (gold: probability 1.0 for the candidates in the document's gold
        set). When per-document source weighting is enabled
        (noul_questions given), it also carries the per-source noul
        questions, with gold P(true)=1 for each source whose own
        suggestions include a gold subject of the document and 0
        otherwise — both keys are set explicitly because Laya's trainer
        defaults a missing noul key to 0.5."""

        suggestions = list(result)
        if not suggestions:
            return None
        option_text, choice_questions = self._choice_question(suggestions, params)
        gold_ids = set(doc.subject_set)
        choice_gold = {
            label: 1.0
            for label, suggestion in option_text.items()
            if suggestion.subject_id in gold_ids
        }
        if not choice_gold:
            return None

        questions = dict(choice_questions)
        gold = {"the_subject": {"probabilities": choice_gold}}
        if noul_questions is not None:
            questions.update(noul_questions)
            for project_id, _ in sources:
                source_ids = {
                    suggestion.subject_id
                    for suggestion in aligned_by_source[project_id][doc_index]
                }
                gold[project_id] = {
                    "probabilities": {
                        "false": 0.0,
                        "true": 1.0 if source_ids & gold_ids else 0.0,
                    }
                }

        input_chars = int(params.get("laya_input_chars", 0))
        text = doc.text[:input_chars] if input_chars > 0 else doc.text
        return {"state": text, "questions": questions, "gold": gold}

    def _patch_laya_train_forward_device(self, laya_train) -> None:
        """Short-term workaround for an upstream laya bug (present in
        0.3.29 and 0.4.18): finetune() runs its pre-training calibration
        forward (calibration_records) BEFORE train_model() moves the
        model to the target device. With multi-question training rows
        the eval split is non-empty, so on MPS the still-CPU model gets
        an MPS batch and crashes with 'Passed CPU tensor to MPS op'.

        Wraps laya.train._forward so the model is on the batch device
        before every forward; a no-op when it already is, on any device.
        Remove once upstream moves the model right after load_checkpoint."""

        if getattr(laya_train, "_annif_forward_device_patched", False):
            return
        orig_forward = getattr(laya_train, "_forward", None)
        if orig_forward is None:
            # unknown laya version without a module-level _forward; nothing
            # to patch (and nothing known to be broken)
            self.debug(
                "laya.train has no _forward to patch, skipping device workaround"
            )
            return

        def _forward(model, batch, device, amp, detach_encoder):
            if next(model.parameters()).device != device:
                model.to(device)
            return orig_forward(model, batch, device, amp, detach_encoder)

        laya_train._forward = _forward
        laya_train._annif_forward_device_patched = True
        self.info(
            "applied Annif workaround for laya train device placement "
            "(upstream bug, MPS multi-question rows)"
        )

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        # jobs parallelizes the training-data generation across document
        # batches; the fine-tuning itself is bound by the Laya forward and
        # backward passes on one device, not by CPU parallelism
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "Cannot train laya project with no documents."
                )
            self.info("preparing Laya fine-tuning data")
            self._write_train_data(corpus, params, jobs)
        else:
            if not os.path.exists(train_path):
                raise NotInitializedException(
                    "train data file {} not found".format(train_path),
                    backend_id=self.backend_id,
                )
            self.info("reusing cached training data from previous run")

        # an empty laya_model means language-based routing at suggest
        # time, but fine-tuning needs one concrete base checkpoint
        base_model = params.get("laya_model") or "multilingual"
        epochs = int(params.get("laya_epochs", self.DEFAULT_PARAMETERS["laya_epochs"]))
        micro_batch = int(
            params.get("laya_micro_batch", self.DEFAULT_PARAMETERS["laya_micro_batch"])
        )
        self.info(
            "fine-tuning Laya checkpoint {} on {} "
            "({} epoch(s), micro batch {})".format(
                base_model, train_path, epochs, micro_batch
            )
        )
        try:
            import laya.train as laya_train
            from laya.train import TrainConfig, finetune
        except ImportError as err:
            raise OperationFailedException(
                "Laya package not available, cannot train laya backend"
            ) from err
        self._patch_laya_train_forward_device(laya_train)

        shuffle = annif.util.boolean(params.get("laya_shuffle_options", False))
        micro_batch = int(
            params.get("laya_micro_batch", self.DEFAULT_PARAMETERS["laya_micro_batch"])
        )
        # token budgets: set to train (and then infer) at a shorter
        # depth than the checkpoint's own; Laya writes whatever is
        # used into the saved config, so inference follows training
        max_len = params.get("laya_max_len")
        head_max_len = params.get("laya_head_max_len")
        freeze_encoder = annif.util.boolean(params.get("laya_freeze_encoder", False))
        config = TrainConfig(
            epochs=epochs,
            loss=params.get("laya_loss", self.DEFAULT_PARAMETERS["laya_loss"]),
            shuffle_options=("choice",) if shuffle else (),
            micro_batch=micro_batch,
            grad_accum=int(
                params.get(
                    "laya_grad_accum", self.DEFAULT_PARAMETERS["laya_grad_accum"]
                )
            ),
            freeze_encoder=freeze_encoder,
            max_len=int(max_len) if max_len else None,
            head_max_len=int(head_max_len) if head_max_len else None,
        )
        output_dir = os.path.join(self.datadir, self.MODEL_DIR)
        try:
            summary = finetune(
                data=train_path,
                model_dir=base_model,
                output_dir=output_dir,
                config=config,
                device=params.get("laya_device", "auto"),
            )
        except Exception as err:
            raise OperationFailedException(
                "Laya fine-tuning failed: {}".format(err)
            ) from err
        self.info(
            "Laya fine-tuning complete: {} item(s), final loss {:.4f}, "
            "checkpoint saved to {}".format(
                summary.get("train_items", 0),
                (
                    summary["epoch_loss"][-1]
                    if summary.get("epoch_loss")
                    else float("nan")
                ),
                output_dir,
            )
        )

    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str) -> LayaOptimizer:
        """Get a hyperparameter optimizer that selects the best
        laya_weight for the given corpus at the configured threshold."""
        return LayaOptimizer(self, corpus, metric)

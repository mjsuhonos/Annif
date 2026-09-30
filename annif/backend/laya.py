"""Ensemble backend that re-scores source suggestions with the Laya
decision model (https://laya.convaiinnovations.com, Hugging Face:
convaiinnovations/laya, PyPI package ``laya``).

Like the plain ensemble backend it merges the suggestions of the
configured source projects, but instead of returning the merged scores
as-is, it asks Laya one "choice" question per document ("Which subject
does this text reference?") with the candidate subjects as options, and
blends each option probability with the candidate's source score. The
pipeline is zero-shot: no model is ever trained.

Parameter documentation, measured results and the tuning workflow
(``annif optimize`` / ``annif hyperopt``) live in docs/laya.md.
Requires the ``laya`` Python package; the checkpoint is downloaded
from Hugging Face on first use.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import annif.eval
import annif.util
from annif.exception import NotSupportedException, OperationFailedException
from annif.suggestion import SuggestionBatch, SubjectSuggestion

from . import hyperopt
from .ensemble import BaseEnsembleBackend

if TYPE_CHECKING:
    from datetime import datetime

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
            merged = self._backend._merge_source_batches(
                batch_by_source, sources, params
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


class LayaBackend(BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that re-scores merged suggestions with Laya by
    asking one choice question whose options are the suggested
    subjects."""

    name = "laya"

    # See docs/laya.md for parameter documentation and measurements.
    DEFAULT_PARAMETERS = {
        # option pool for the choice question (and cap on suggestions)
        "limit": 20,
        # instruction of the choice question
        "question_template": "Which subject does this text reference?",
        # blend weight: score = w * option_probability + (1 - w) * source
        "laya_weight": 0.25,
        # minimum returned score, applied inside the backend
        "threshold": 0.0,
        # checkpoint; set to an empty value to let Laya route by language
        "laya_model": "multilingual",
        # cap on the raw text sent to Laya (tokenizer savings)
        "laya_input_chars": 0,
    }

    # class-level default so uninitialized instances behave like other
    # backends (pattern from MLLMBackend._model)
    _router = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        if self._router is None:
            self.info("loading Laya decision model (downloads checkpoint on first use)")
            try:
                # imported lazily so Annif runs without the package
                from laya import Router
            except ImportError as err:
                raise OperationFailedException(
                    "Laya package not available, cannot use laya backend"
                ) from err
            self._router = Router()
            self.info("Laya decision model loaded")

    def _subject_label(self, subject: Subject) -> str:
        """Best available label for a subject: the vocabulary language
        first, then any label, then the URI."""
        labels = subject.labels or {}
        label = labels.get(self.project.vocab_lang)
        if not label and labels:
            label = next(iter(labels.values()))
        return label or subject.uri

    def _predict_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        """Optional Router.predict() keyword arguments from the backend
        parameters: checkpoint (laya_model) and token budgets
        (laya_max_len, laya_head_max_len)."""
        kwargs = {}
        if params.get("laya_model"):
            kwargs["model"] = params["laya_model"]
        if params.get("laya_max_len"):
            kwargs["max_len"] = int(params["laya_max_len"])
        if params.get("laya_head_max_len"):
            kwargs["head_max_len"] = int(params["laya_head_max_len"])
        return kwargs

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

        default_template = self.DEFAULT_PARAMETERS["question_template"]
        instructions = params.get("question_template", default_template)
        input_chars = int(params.get("laya_input_chars", 0))
        predict_kwargs = self._predict_kwargs(params)

        verified = []
        for doc, result in zip(documents, merged):
            suggestions = list(result)

            if not suggestions:
                verified.append(([], "empty"))
                continue

            # option texts must be unique: duplicate labels get a
            # counter suffix (Laya answers are keyed by these texts)
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
        the source scores, then apply the output limit and threshold."""

        laya_weight = float(
            params.get("laya_weight", self.DEFAULT_PARAMETERS["laya_weight"])
        )
        threshold = float(params.get("threshold") or 0.0)
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
        merged = self._merge_source_batches(batch_by_source, sources, params)
        return self._rescore_with_laya(documents, merged, params)

    @property
    def is_trained(self) -> bool:
        # trained when all source projects are trained; Laya itself is
        # a pretrained, non-trainable component
        return all(self._get_sources_attribute("is_trained"))

    @property
    def modification_time(self) -> datetime | None:
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        # train the source projects instead; this backend has no
        # trainable model of its own
        raise NotSupportedException("Training laya backend is not possible.")

    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str) -> LayaOptimizer:
        """Get a hyperparameter optimizer that selects the best
        laya_weight for the given corpus at the configured threshold."""
        return LayaOptimizer(self, corpus, metric)

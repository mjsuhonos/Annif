"""Ensemble backend that re-scores source suggestions with the Laya
decision model (https://laya.convaiinnovations.com, Hugging Face:
convaiinnovations/laya, PyPI package ``laya``).

Like the plain ensemble backend it merges the suggestions of the
configured source projects, but instead of returning the merged scores
as-is, it asks Laya about the candidate subjects (one choice question
per document, one yes/no proposition per candidate, or both, depending
on the laya_score_mode parameter) and blends the answers with the
source scores. The blend follows the CLM backend: both score sets are
min-max normalized per document (the blend-source parameter can keep
the source scores raw) and a missing Laya score counts as 0.0. The
pipeline is zero-shot: no model is ever trained.

Parameter documentation and the tuning workflow
(``annif hyperopt``) live in docs/laya.md.
Requires the ``laya`` Python package; the checkpoint is downloaded
from Hugging Face on first use.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import annif.eval
import annif.util
from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SuggestionBatch, SubjectSuggestion

from . import hyperopt
from .ensemble import BaseEnsembleBackend

if TYPE_CHECKING:
    from datetime import datetime

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.corpus.document import Document, DocumentCorpus
    from annif.vocab.types import Subject


def _parse_score_mode(mode: str) -> frozenset[str]:
    """Parse the laya_score_mode parameter into the set of answer kinds
    it names: "choice", "noul", or both ("choice,noul" and "noul,choice"
    are equivalent)."""
    kinds = frozenset(kind.strip() for kind in mode.lower().split(","))
    if not kinds <= {"choice", "noul"}:
        raise ConfigurationException(
            "laya_score_mode must be 'choice', 'noul' or a combination "
            "of both, got '{}'".format(mode)
        )
    return kinds


def _parse_blend_source(blend_source: str) -> str:
    """Validate the blend-source parameter: "normalized" (the
    CLM-style default) or "raw"."""
    if blend_source not in ("normalized", "raw"):
        raise ConfigurationException(
            "blend-source must be 'normalized' or 'raw', "
            "got '{}'".format(blend_source)
        )
    return blend_source


def _blend(
    suggestions: list[SubjectSuggestion],
    laya_scores: dict[str, dict[int, float | None]],
    kinds: frozenset[str],
    alpha: float,
    blend_source: str = "normalized",
    debug=None,
) -> list[SubjectSuggestion]:
    """Blend the source and Laya scores of one document, keeping all
    candidates. The Laya scores are min-max normalized per document (a
    degenerate set maps to 0.5), and the new score is
    alpha * source + (1 - alpha) * laya_norm, where the source scores
    are either min-max normalized per document like the Laya scores
    ("normalized", the CLM-style default) or used as-is ("raw"),
    keeping their absolute confidence.

    The Laya value of a candidate is the answer of the single requested
    kind, or the average of the available answers when both kinds are
    requested; a candidate with no answer at all counts as 0.0, and the
    misses are reported through the given debug logger, if any."""

    def norm(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.5
        return (value - lo) / (hi - lo)

    values = []
    missing = []
    for suggestion in suggestions:
        answers = [
            laya_scores[kind][suggestion.subject_id]
            for kind in sorted(kinds)
            if laya_scores.get(kind, {}).get(suggestion.subject_id) is not None
        ]
        if answers:
            values.append(sum(answers) / len(answers))
        else:
            # a candidate without any Laya answer counts as 0.0
            values.append(0.0)
            missing.append(suggestion.subject_id)
    if missing and debug is not None:
        debug(
            "missing Laya score for subject(s) {}, counting as 0.0".format(
                ", ".join(str(subject_id) for subject_id in missing)
            )
        )

    if blend_source == "raw":
        source_values = [suggestion.score for suggestion in suggestions]
    else:
        src_min = min(suggestion.score for suggestion in suggestions)
        src_max = max(suggestion.score for suggestion in suggestions)
        source_values = [
            norm(suggestion.score, src_min, src_max) for suggestion in suggestions
        ]
    lo, hi = min(values), max(values)

    blended = [
        SubjectSuggestion(
            subject_id=suggestion.subject_id,
            score=alpha * source_value + (1.0 - alpha) * norm(laya_value, lo, hi),
        )
        for suggestion, source_value, laya_value in zip(
            suggestions, source_values, values
        )
    ]
    # the blend can change the ranking, so sort explicitly
    blended.sort(key=lambda suggestion: suggestion.score, reverse=True)
    return blended


def _format_duration(seconds: float) -> str:
    """Format a rough remaining duration for progress logging."""
    if seconds >= 60:
        return "~{:.0f} min".format(seconds / 60)
    return "~{:.0f}s".format(seconds)


class LayaHPObjective(hyperopt.HPObjective):
    """Objective of the laya hyperparameter optimizer: samples
    blend-alpha and blend-source - and laya_score_mode, when the cached
    answers cover more than one mode - and re-scores the Laya answers
    cached by _prepare() exactly like the backend does at suggest time
    - at the configured limit - so trials need no model inference."""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        alpha = trial.suggest_float("blend-alpha", 0.0, 1.0)
        blend_source = trial.suggest_categorical("blend-source", ["normalized", "raw"])
        modes = args["modes"]
        if len(modes) > 1:
            mode = trial.suggest_categorical("laya_score_mode", modes)
            kinds = _parse_score_mode(mode)
        else:
            kinds = _parse_score_mode(modes[0])
        limit = args["limit"]

        eval_batch = annif.eval.EvaluationBatch(args["subject_index"])
        for verified, gold_batch in zip(args["verified_batches"], args["gold_batches"]):
            rescored_results = []
            for suggestions, laya_scores, status in verified:
                if status == "empty":
                    rescored_results.append([])
                elif status == "failed":
                    # no usable answer: keep the source scores
                    rescored_results.append(list(suggestions))
                else:
                    rescored_results.append(
                        _blend(suggestions, laya_scores, kinds, alpha, blend_source)
                    )
            batch = SuggestionBatch.from_sequence(
                rescored_results, args["subject_index"]
            )
            batch = batch.filter(limit=limit)
            eval_batch.evaluate_many(batch, gold_batch)

        return eval_batch.results(metrics=[args["metric"]])[args["metric"]]


class LayaOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the laya backend: selects
    blend-alpha and blend-source - and laya_score_mode, when the
    configured mode is "choice,noul" - for a labeled corpus, at the
    limit configured in the project. _prepare() runs the source merge
    and asks Laya the question kinds of the configured mode once; the
    trials re-score the cached answers, so any number of trials costs
    one inference pass over the corpus."""

    def __init__(
        self,
        backend: LayaBackend,
        corpus: DocumentCorpus,
        metric: str,
    ) -> None:
        super().__init__(backend, corpus, metric, LayaHPObjective)

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        # n_jobs is unused: preparation is bound by the single Laya
        # request per document, not by CPU parallelism
        self._backend.initialize()
        params = self._backend._get_backend_params(None)
        sources = annif.util.parse_sources(params["sources"])

        # ask the question kinds of the configured mode; the trials can
        # compare only the modes whose kinds are within them, so the
        # cross-mode comparison is bought by configuring "choice,noul"
        kinds = _parse_score_mode(params.get("laya_score_mode", "choice"))
        modes = [
            mode
            for mode in ("choice", "noul", "choice,noul")
            if _parse_score_mode(mode) <= kinds
        ]

        # count the documents first (corpora are re-iterable, like in
        # DocumentCorpus.is_empty) so that the caching pass can report
        # progress
        total_docs = sum(1 for _ in self._corpus.documents)
        self._backend.info(
            "Laya: caching answers for {} doc(s) ({})".format(
                total_docs, " + ".join(sorted(kinds))
            )
        )

        verified_batches = []
        gold_batches = []
        docs_done = 0
        questions = 0
        started = time.monotonic()
        for doc_batch in self._corpus.doc_batches:
            docs = list(doc_batch)
            batch_by_source = self._backend._suggest_with_sources(docs, sources)
            merged = self._backend._merge_source_batches(
                batch_by_source, sources, params
            )
            verified = self._backend._verify_batch(docs, merged, params)
            verified_batches.append(verified)
            gold_batches.append([doc.subject_set for doc in docs])

            docs_done += len(docs)
            questions += sum(
                (1 if "choice" in kinds else 0)
                + (len(suggestions) if "noul" in kinds else 0)
                for suggestions, _laya_scores, status in verified
                if status != "empty"
            )
            elapsed = time.monotonic() - started
            progress = (
                "Laya: cached {}/{} doc(s) ({:.1f}%), {} question(s), "
                "{:.1f}s".format(
                    docs_done,
                    total_docs,
                    100.0 * docs_done / total_docs,
                    questions,
                    elapsed,
                )
            )
            if docs_done < total_docs:
                remaining = (total_docs - docs_done) * elapsed / docs_done
                progress += ", {} left".format(_format_duration(remaining))
            self._backend.info(progress)

        return {
            "verified_batches": verified_batches,
            "gold_batches": gold_batches,
            "subject_index": self._backend.project.subjects,
            "limit": int(params["limit"]),
            "metric": self._metric,
            "modes": modes,
        }

    def _postprocess(self, study: Study) -> hyperopt.HPRecommendation:
        lines = [
            "blend-alpha = {:.4f}".format(study.best_params["blend-alpha"]),
            "blend-source = {}".format(study.best_params["blend-source"]),
        ]
        if "laya_score_mode" in study.best_params:
            lines.append(
                "laya_score_mode = {}".format(study.best_params["laya_score_mode"])
            )
        return hyperopt.HPRecommendation(lines=lines, score=study.best_value)


class LayaBackend(BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that re-scores merged suggestions with Laya:
    one choice question over the candidates, one yes/no proposition per
    candidate, or both - blended CLM-style, with the source scores
    min-max normalized per document or used raw."""

    name = "laya"

    # See docs/laya.md for parameter documentation.
    DEFAULT_PARAMETERS = {
        # candidate pool: the top-limit merged candidates are compared
        # (and the cap on returned suggestions)
        "limit": 20,
        # instruction of the choice question
        "question_template": "Which subject does this document reference?",
        # proposition of the noul questions; {label} is required
        "instruction": "This document is about {label}.",
        # scoring mode: which questions Laya is asked and which answers
        # score the candidates - "choice", "noul", or "choice,noul"
        # for both
        "laya_score_mode": "choice",
        # blend weight: score = alpha * source
        # + (1 - alpha) * norm(laya)
        "blend-alpha": 0.85,
        # how the source scores enter the blend: min-max normalized
        # per document ("normalized", the CLM-style default) or as-is
        # ("raw", keeping their absolute confidence)
        "blend-source": "normalized",
        # checkpoint; default to an empty value to let Laya route by language
        "laya_model": "",
        # cap on the raw text sent to Laya (tokenizer savings)
        "laya_input_chars": 0,
        # optional Laya token budgets
        "laya_max_len": 0,
        "laya_head_max_len": 0,
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

    def _build_questions(
        self,
        suggestions: list[SubjectSuggestion],
        params: dict[str, Any],
        kinds: frozenset[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, SubjectSuggestion]]:
        """Build the Laya questions of one document, all asked together
        in a single request: a choice question over the candidate labels
        ("choice"), one yes/no proposition per candidate ("noul"), or
        both. Returns the questions and a mapping of choice option text
        to suggestion (empty when no choice question is asked)."""

        questions = {}
        option_text = {}

        if "choice" in kinds:
            # option texts must be unique: duplicate labels get a
            # counter suffix (Laya answers are keyed by these texts)
            for suggestion in suggestions:
                subject = self.project.subjects[suggestion.subject_id]
                base = self._subject_label(subject)
                label, n = base, 1
                while label in option_text:
                    n += 1
                    label = "{} ({})".format(base, n)
                option_text[label] = suggestion

            questions["the_subject"] = {
                "type": "choice",
                "instructions": params.get(
                    "question_template", self.DEFAULT_PARAMETERS["question_template"]
                ),
                "criteria": {label: "" for label in option_text},
            }

        if "noul" in kinds:
            instruction = params.get(
                "instruction", self.DEFAULT_PARAMETERS["instruction"]
            )
            if "{label}" not in instruction:
                raise ConfigurationException(
                    "instruction parameter must contain a {label} placeholder"
                )
            for suggestion in suggestions:
                subject = self.project.subjects[suggestion.subject_id]
                label = self._subject_label(subject)
                questions[str(suggestion.subject_id)] = {
                    "type": "noul",
                    "instructions": instruction.format(label=label),
                }

        return questions, option_text

    def _verify_batch(
        self,
        documents: list[Document],
        merged: SuggestionBatch,
        params: dict[str, Any],
    ) -> list[tuple[list[SubjectSuggestion], dict[str, dict[int, float | None]], str]]:
        """Ask Laya to verify the merged candidates of each document in
        a single request, asking the questions of the configured
        laya_score_mode.

        Returns one (suggestions, laya_scores, status) entry per
        document. laya_scores holds, per asked kind, a
        subject_id -> score mapping where an unanswered question is
        None (counted as 0.0 at scoring time). status is "ok",
        "failed" or "empty". This is the expensive half of the backend
        (one Laya request per document); the hyperparameter optimizer
        reuses it to cache the answers once for all its trials, so
        every scoring experiment runs on the cached answers."""

        kinds = _parse_score_mode(params.get("laya_score_mode", "choice"))
        input_chars = int(params.get("laya_input_chars", 0))

        verified = []
        for doc_idx, (doc, result) in enumerate(zip(documents, merged), 1):
            suggestions = list(result)
            if not suggestions:
                verified.append(([], {}, "empty"))
                continue

            questions, option_text = self._build_questions(suggestions, params, kinds)

            # laya_input_chars: the sources always see the full text,
            # only the questions may use the truncated text
            text = doc.text[:input_chars] if input_chars > 0 else doc.text

            self.debug(
                "asking Laya {} question(s) ({} choice + {} noul) for "
                "document {}/{} of {} characters".format(
                    len(questions),
                    1 if "choice" in kinds else 0,
                    len(suggestions) if "noul" in kinds else 0,
                    doc_idx,
                    len(documents),
                    len(doc.text),
                )
            )

            try:
                answer = self._router.predict(
                    text, questions, **self._predict_kwargs(params)
                )
            except Exception as err:
                # keep the plain ensemble scores rather than failing the
                # whole suggestion batch
                self.warning(
                    "Laya prediction failed, keeping original scores: {}".format(err)
                )
                verified.append((suggestions, {}, "failed"))
                continue

            answers = answer.get("answers", {})
            laya_scores = {}

            if "choice" in kinds:
                probabilities = (
                    answers.get("the_subject", {}).get("probabilities") or {}
                )
                choice_scores = {}
                for label, suggestion in option_text.items():
                    probability = probabilities.get(label)
                    choice_scores[suggestion.subject_id] = (
                        float(probability) if probability is not None else None
                    )
                laya_scores["choice"] = choice_scores

            if "noul" in kinds:
                noul_scores = {}
                for suggestion in suggestions:
                    value = answers.get(str(suggestion.subject_id), {}).get("noul")
                    noul_scores[suggestion.subject_id] = (
                        float(value) if value is not None else None
                    )
                laya_scores["noul"] = noul_scores

            verified.append((suggestions, laya_scores, "ok"))

        return verified

    def _rescore_with_laya(
        self,
        documents: list[Document],
        merged: SuggestionBatch,
        params: dict[str, Any],
    ) -> SuggestionBatch:
        """Blend the cached Laya scores (from _verify_batch) with the
        source scores, then apply the output limit."""

        alpha = float(params.get("blend-alpha", self.DEFAULT_PARAMETERS["blend-alpha"]))
        blend_source = _parse_blend_source(
            params.get("blend-source", self.DEFAULT_PARAMETERS["blend-source"])
        )
        kinds = _parse_score_mode(params.get("laya_score_mode", "choice"))

        started = time.monotonic()

        verified = self._verify_batch(documents, merged, params)

        rescored_results = []
        candidates = 0
        for suggestions, laya_scores, status in verified:
            if status == "empty":
                rescored_results.append([])
                continue

            candidates += len(suggestions)

            if status == "failed":
                # no usable answer: keep the source scores
                rescored_results.append(list(suggestions))
                continue

            rescored_results.append(
                _blend(
                    suggestions,
                    laya_scores,
                    kinds,
                    alpha,
                    blend_source,
                    debug=self.debug,
                )
            )

        self.info(
            "Laya: {} doc(s), {} candidate(s), {:.2f}s".format(
                len(documents), candidates, time.monotonic() - started
            )
        )

        # filter after building the rows: the output must keep the
        # top-limit candidates by blended score, not by the source
        # ranking they arrived in
        batch = SuggestionBatch.from_sequence(rescored_results, self.project.subjects)
        return batch.filter(limit=int(params["limit"]))

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
        blend-alpha and blend-source - and laya_score_mode, when the
        configured mode allows comparing modes - for the given corpus."""
        return LayaOptimizer(self, corpus, metric)

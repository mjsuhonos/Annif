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

import json
import os.path
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import annif.eval
import annif.util
from annif.exception import (
    NotInitializedException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SuggestionBatch, SubjectSuggestion

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
    subjects. Without training the pipeline is zero-shot; annif train
    fine-tunes the checkpoint on a labeled corpus."""

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
    ) -> None:
        """Write the Laya fine-tuning data for the corpus: a JSONL file
        of {state, questions, gold} rows, one row per document whose
        gold subjects include at least one of the merged candidates.
        Each row carries exactly the choice question the backend asks at
        suggest time, so the fine-tuned checkpoint sees the same
        question distribution it will be answering."""

        sources = annif.util.parse_sources(params["sources"])
        for project_id, _ in sources:
            self.project.registry.get_project(project_id).initialize()

        input_chars = int(params.get("laya_input_chars", 0))
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        rows = 0
        skipped = 0
        with open(train_path, "w", encoding="utf-8") as train_file:
            for doc_batch in corpus.doc_batches:
                docs = list(doc_batch)
                batch_by_source = self._suggest_with_sources(docs, sources)
                merged = self._merge_source_batches(batch_by_source, sources, params)
                for doc, result in zip(docs, merged):
                    probabilities = {}
                    questions = None
                    suggestions = list(result)
                    if suggestions:
                        option_text, questions = self._choice_question(
                            suggestions, params
                        )
                        gold_ids = set(doc.subject_set)
                        probabilities = {
                            label: 1.0
                            for label, suggestion in option_text.items()
                            if suggestion.subject_id in gold_ids
                        }
                    if not probabilities:
                        # Laya cannot learn from a question whose answer
                        # is not among the options
                        skipped += 1
                        continue
                    text = doc.text[:input_chars] if input_chars > 0 else doc.text
                    row = {
                        "state": text,
                        "questions": questions,
                        "gold": {"the_subject": {"probabilities": probabilities}},
                    }
                    print(json.dumps(row, ensure_ascii=False), file=train_file)
                    rows += 1
        self.info(
            "wrote {} Laya fine-tuning row(s) to {}, skipped {} document(s) "
            "without a gold candidate".format(rows, train_path, skipped)
        )
        if rows == 0:
            raise NotSupportedException(
                "no document had a gold subject among the merged candidates"
            )

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        # jobs is unused: fine-tuning is bound by the Laya forward and
        # backward passes on one device, not by CPU parallelism
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "Cannot train laya project with no documents."
                )
            self.info("preparing Laya fine-tuning data")
            self._write_train_data(corpus, params)
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
            from laya.train import TrainConfig, finetune
        except ImportError as err:
            raise OperationFailedException(
                "Laya package not available, cannot train laya backend"
            ) from err

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

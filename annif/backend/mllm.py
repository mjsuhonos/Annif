"""Maui-like Lexical Matching backend"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

import annif.eval
import annif.parallel
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.lexical.mllm import (
    Feature,
    MLLMModel,
    create_classifier,
    prediction_to_list,
)
from annif.suggestion import vector_to_suggestions

from . import hyperopt

if TYPE_CHECKING:
    from collections.abc import Iterator

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus import Document, DocumentCorpus
    from annif.lexical.mllm import CandidateBlock
    from annif.vocab import SubjectIndex


def prediction_to_result(
    prediction: list[tuple[np.float64, int]],
    params: dict[str, Any],
    subject_index: SubjectIndex,
) -> Iterator:
    vector = np.zeros(len(subject_index), dtype=np.float32)
    for score, subject_id in prediction:
        vector[subject_id] = score
    return vector_to_suggestions(vector, int(params["limit"]))


class MLLMHPObjective(hyperopt.HPObjective):
    """Objective function of the MLLM hyperparameter optimizer"""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        params = {
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 100, 2000),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "limit": 100,
        }
        model = create_classifier(params)
        model.fit(args["train_x"], args["train_y"])

        # score the candidates of all documents in a single predict_proba
        # call, then split the probabilities per document; the rows of the
        # precomputed eval feature matrix are in corpus document order
        proba = model.predict_proba(args["features"])

        batch = annif.eval.EvaluationBatch(args["subject_index"])
        start = 0
        for goldsubj, candidates in zip(args["gold_subjects"], args["candidates"]):
            end = start + len(candidates.subject_ids)
            if end > start:
                scores = proba[start:end]
                ranking = prediction_to_list(scores, candidates.subject_ids)
            else:
                ranking = []
            results = prediction_to_result(ranking, params, args["subject_index"])
            batch.evaluate_many([results], [goldsubj])
            start = end
        results = batch.results(metrics=[args["metric"]])
        return results[args["metric"]]


class MLLMCandidateGenerator(annif.parallel.BaseWorker):
    """Worker class for generating candidates in parallel"""

    @classmethod
    def generate_for_doc_batch(cls, doc_batch) -> tuple[list, list]:
        """Generate candidates for a batch of documents"""
        backend = cls.args["backend"]
        candidates = [backend._generate_candidates(doc.text) for doc in doc_batch]
        subject_sets = [doc.subject_set for doc in doc_batch]
        return (candidates, subject_sets)


class MLLMOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the MLLM backend"""

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        self._backend.initialize()
        train_x, train_y = self._backend._load_train_data()
        all_candidates = []
        gold_subjects = []

        if n_jobs == 1:
            for doc in self._corpus.documents:
                all_candidates.append(self._backend._generate_candidates(doc.text))
                gold_subjects.append(doc.subject_set)
        else:
            n_jobs, pool_constructor = annif.parallel.get_pool(n_jobs)
            with pool_constructor(
                n_jobs,
                initializer=MLLMCandidateGenerator.init,
                initargs=({"backend": self._backend},),
            ) as pool:
                # ordered map: the candidate blocks must arrive in corpus
                # order so that they stay aligned with the gold subjects and
                # the rows of the precomputed eval feature matrix
                for candidates, subject_sets in pool.imap(
                    MLLMCandidateGenerator.generate_for_doc_batch,
                    self._corpus.doc_batches,
                ):
                    all_candidates.extend(candidates)
                    gold_subjects.extend(subject_sets)

        # convert the candidates of the hyperopt corpus to feature values
        # once, so that every trial can score the eval corpus without
        # recomputing the features
        features = np.zeros(
            (
                sum(len(c.subject_ids) for c in all_candidates),
                len(Feature),
            ),
            dtype=np.float32,
        )
        self._backend._model._prepare_features(all_candidates, features, n_jobs)

        return {
            "train_x": train_x,
            "train_y": train_y,
            "subject_index": self._backend.project.subjects,
            "gold_subjects": gold_subjects,
            "candidates": all_candidates,
            "features": features,
            "metric": self._metric,
        }

    def _postprocess(self, study: Study) -> HPRecommendation:
        bp = study.best_params
        lines = [
            f"min_samples_leaf={bp['min_samples_leaf']}",
            f"max_leaf_nodes={bp['max_leaf_nodes']}",
            f"max_samples={bp['max_samples']:.4f}",
        ]
        return hyperopt.HPRecommendation(lines=lines, score=study.best_value)


class MLLMBackend(hyperopt.AnnifHyperoptBackend):
    """Maui-like Lexical Matching backend for Annif"""

    name = "mllm"

    # defaults for unitialized instances
    _model = None

    MODEL_FILE = "mllm-model.gz"
    TRAIN_FILE = "mllm-train.gz"

    DEFAULT_PARAMETERS = {
        "min_samples_leaf": 20,
        "max_leaf_nodes": 1000,
        "max_samples": 0.9,
        "use_hidden_labels": False,
    }

    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str) -> MLLMOptimizer:
        return MLLMOptimizer(self, corpus, metric, MLLMHPObjective)

    def _load_model(self) -> MLLMModel:
        path = os.path.join(self.datadir, self.MODEL_FILE)
        self.debug("loading model from {}".format(path))
        if os.path.exists(path):
            return MLLMModel.load(path)
        else:
            raise NotInitializedException(
                "model {} not found".format(path), backend_id=self.backend_id
            )

    def _load_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.datadir, self.TRAIN_FILE)
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise NotInitializedException(
                "train data file {} not found".format(path), backend_id=self.backend_id
            )

    def initialize(self, parallel: bool = False) -> None:
        if self._model is None:
            self._model = self._load_model()

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        self.info("starting train")
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "training backend {} with no documents".format(self.backend_id)
                )
            self.info("preparing training data")
            self._model = MLLMModel()
            train_data = self._model.prepare_train(
                corpus, self.project.vocab, self.project.analyzer, params, jobs
            )
            annif.util.atomic_save(
                train_data, self.datadir, self.TRAIN_FILE, method=joblib.dump
            )
        else:
            self.info("reusing cached training data from previous run")
            self._model = self._load_model()
            train_data = self._load_train_data()

        self.info("training model")
        self._model.train(train_data[0], train_data[1], params, n_jobs=jobs or None)

        self.info("saving model")
        annif.util.atomic_save(self._model, self.datadir, self.MODEL_FILE)

    def _generate_candidates(self, text: str) -> CandidateBlock:
        return self._model.generate_candidates(text, self.project.analyzer)

    def _suggest(self, doc: Document, params: dict[str, Any]) -> Iterator:
        candidates = self._generate_candidates(doc.text)
        prediction = self._model.predict(candidates)
        return prediction_to_result(prediction, params, self.project.subjects)

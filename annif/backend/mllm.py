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
    candidates_to_features,
    create_classifier,
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
    scores: np.ndarray,
    subject_ids: np.ndarray,
    params: dict[str, Any],
    subject_index: SubjectIndex,
) -> Iterator:
    vector = np.zeros(len(subject_index), dtype=np.float32)
    vector[subject_ids] = scores
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
        for goldsubj, subject_ids in zip(args["gold_subjects"], args["candidates"]):
            end = start + len(subject_ids)
            results = prediction_to_result(
                proba[start:end, 1], subject_ids, params, args["subject_index"]
            )
            batch.evaluate_many([results], [goldsubj])
            start = end
        results = batch.results(metrics=[args["metric"]])
        return results[args["metric"]]


class MLLMCandidateGenerator(annif.parallel.BaseWorker):
    """Worker class for generating candidates and converting them to
    feature values in parallel"""

    @classmethod
    def generate_and_convert_for_doc_batch(cls, doc_batch):
        """Generate candidates for a batch of documents and convert them
        to feature values. Returns a tuple of the concatenated feature
        matrix for the batch, the per-document candidate subject id
        arrays and the per-document gold subject sets."""
        backend = cls.args["backend"]
        mdata = cls.args["mdata"]
        doc_features = []
        subject_ids_list = []
        subject_sets = []
        for doc in doc_batch:
            candidates = backend._generate_candidates(doc.text)
            doc_features.append(candidates_to_features(candidates, mdata))
            subject_ids_list.append(candidates.subject_ids)
            subject_sets.append(doc.subject_set)
        return (np.concatenate(doc_features), subject_ids_list, subject_sets)


class MLLMOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the MLLM backend"""

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        self._backend.initialize()
        train_x, train_y = self._backend._load_train_data()
        mdata = self._backend._model._model_data
        all_subject_ids = []
        gold_subjects = []
        feature_chunks = []

        if n_jobs == 1:
            for doc in self._corpus.documents:
                candidates = self._backend._generate_candidates(doc.text)
                feature_chunks.append(candidates_to_features(candidates, mdata))
                all_subject_ids.append(candidates.subject_ids)
                gold_subjects.append(doc.subject_set)
        else:
            n_jobs, pool_constructor = annif.parallel.get_pool(n_jobs)
            with pool_constructor(
                n_jobs,
                initializer=MLLMCandidateGenerator.init,
                initargs=({"backend": self._backend, "mdata": mdata},),
            ) as pool:
                # ordered map: results must arrive in corpus order so that
                # the subject ids, gold subjects and feature rows align
                for features, subject_ids_list, subject_sets in pool.imap(
                    MLLMCandidateGenerator.generate_and_convert_for_doc_batch,
                    self._corpus.doc_batches,
                ):
                    feature_chunks.append(features)
                    all_subject_ids.extend(subject_ids_list)
                    gold_subjects.extend(subject_sets)

        # assemble the eval feature matrix in one preallocated array; rows
        # are in corpus document order and match the subject id arrays
        n_candidates = sum(len(subject_ids) for subject_ids in all_subject_ids)
        features = np.zeros((n_candidates, len(Feature)), dtype=np.float32)
        start = 0
        for chunk_idx, chunk in enumerate(feature_chunks):
            end = start + len(chunk)
            features[start:end] = chunk
            feature_chunks[chunk_idx] = None
            start = end

        return {
            "train_x": train_x,
            "train_y": train_y,
            "subject_index": self._backend.project.subjects,
            "gold_subjects": gold_subjects,
            "candidates": all_subject_ids,
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
        scores = self._model.predict(candidates)
        return prediction_to_result(
            scores, candidates.subject_ids, params, self.project.subjects
        )

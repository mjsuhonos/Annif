"""Unit tests for the MLLM backend in Annif"""

import multiprocessing.dummy

import numpy as np
import pytest

import annif
import annif.backend
import annif.parallel
from annif.backend.mllm import MLLMCandidateGenerator
from annif.lexical.mllm import Feature, candidates_to_features
from annif.corpus import Document
from annif.exception import NotInitializedException, NotSupportedException


def test_mllm_default_params(project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(backend_id="mllm", config_params={}, project=project)

    expected_default_params = {
        "limit": 100,  # from AnnifBackend class
        "min_samples_leaf": 20,
        "max_leaf_nodes": 1000,
        "max_samples": 0.9,
    }
    actual_params = mllm.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_mllm_train(datadir, fulltext_corpus, project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )

    mllm.train(fulltext_corpus)
    assert mllm._model is not None
    assert datadir.join("mllm-train.gz").exists()
    assert datadir.join("mllm-train.gz").size() > 0
    assert datadir.join("mllm-model.gz").exists()
    assert datadir.join("mllm-model.gz").size() > 0


def test_mllm_train_cached(datadir, project):
    modelfile = datadir.join("mllm-model.gz")
    assert modelfile.exists()

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )

    mllm.train("cached")
    assert mllm._model is not None
    assert modelfile.exists()
    assert modelfile.size() > 0
    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_mllm_train_nodocuments(project, empty_corpus):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )

    with pytest.raises(NotSupportedException) as excinfo:
        mllm.train(empty_corpus)
    assert "training backend mllm with no documents" in str(excinfo.value)


def test_mllm_suggest(project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm", config_params={"limit": 8, "language": "fi"}, project=project
    )

    results = mllm.suggest(
        [
            Document(
                text="""Arkeologia on tieteenala, jota sanotaan joskus
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
            )
        ]
    )[0]

    assert len(results) > 0
    assert len(results) <= 8
    archaeology = project.subjects.by_uri("http://www.yso.fi/onto/yso/p1265")
    assert archaeology in [result.subject_id for result in results]


def test_mllm_suggest_no_matches(project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm", config_params={"limit": 8, "language": "fi"}, project=project
    )

    results = mllm.suggest([Document(text="Nothing matches this.")])[0]

    assert len(results) == 0


def test_mllm_hyperopt(project, fulltext_corpus):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )

    optimizer = mllm.get_hp_optimizer(fulltext_corpus, metric="NDCG")
    optimizer.optimize(n_trials=3, n_jobs=1, results_file=None)


def test_mllm_hyperopt_eval_corpus_differs_from_train(
    project, fulltext_corpus, document_corpus
):
    """Regression test: the hyperopt corpus may differ from the corpus the
    backend was trained with, so the objective must score candidates
    generated from the hyperopt corpus (via the precomputed feature
    matrix), not the cached training rows"""
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )
    mllm.train(fulltext_corpus)

    optimizer = mllm.get_hp_optimizer(document_corpus, metric="NDCG")
    args = optimizer._prepare(n_jobs=1)

    # the feature matrix rows must match the eval corpus's candidates
    n_candidates = sum(len(subject_ids) for subject_ids in args["candidates"])
    assert args["features"].shape == (n_candidates, len(Feature))
    # eval corpus differs from the training corpus (different docs)
    assert args["features"].shape[0] != args["train_x"].shape[0]
    # feature rows must be derived from the eval corpus's candidates
    first = list(document_corpus.documents)[0]
    first_block = mllm._generate_candidates(first.text)
    assert np.array_equal(
        args["features"][: len(first_block.subject_ids)],
        candidates_to_features(first_block, mllm._model._model_data),
    )

    result = optimizer.optimize(n_trials=2, n_jobs=1, results_file=None)
    assert len(result.lines) == 3


def test_mllm_candidate_generator_doc_batch(project, fulltext_corpus):
    """Test the worker method for generating candidates and converting
    them to feature values for a batch of documents, returning the batch
    feature matrix, subject id arrays and subject sets"""
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )
    mllm.initialize()
    MLLMCandidateGenerator.init(
        {"backend": mllm, "mdata": mllm._model._model_data}
    )

    doc_batch = list(fulltext_corpus.documents)[:4]
    features, subject_ids_list, subject_sets = (
        MLLMCandidateGenerator.generate_and_convert_for_doc_batch(doc_batch)
    )

    assert len(subject_ids_list) == len(doc_batch)
    assert len(subject_sets) == len(doc_batch)
    assert subject_sets == [doc.subject_set for doc in doc_batch]
    assert features.shape == (
        sum(len(subject_ids) for subject_ids in subject_ids_list),
        len(Feature),
    )
    assert features.dtype == np.float32
    assert features.shape[0] > 0


def test_mllm_hyperopt_parallel_n_jobs_2(project, fulltext_corpus, monkeypatch):
    """Test MLLM hyperopt with n_jobs=2, processing the corpus in multiple
    document batches and accumulating their unordered results"""
    monkeypatch.setattr(fulltext_corpus, "DOC_BATCH_SIZE", 4)
    monkeypatch.setattr(
        annif.parallel, "get_pool", lambda n_jobs: (n_jobs, multiprocessing.dummy.Pool)
    )
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )

    optimizer = mllm.get_hp_optimizer(fulltext_corpus, metric="NDCG")
    serial_args = optimizer._prepare(n_jobs=1)
    parallel_args = optimizer._prepare(n_jobs=2)

    def cand_key(subject_ids):
        """hashable representation of a candidate subject id array"""
        return tuple(subject_ids.tolist())

    def paired_results(args):
        return sorted(
            (cand_key(candidates), tuple(sorted(subjects)))
            for candidates, subjects in zip(args["candidates"], args["gold_subjects"])
        )

    assert len(parallel_args["candidates"]) == len(serial_args["candidates"])
    assert len(parallel_args["gold_subjects"]) == len(serial_args["gold_subjects"])
    assert paired_results(parallel_args) == paired_results(serial_args)
    # the precomputed feature matrices must be identical, row for row;
    # this also checks that parallel candidates arrive in corpus order
    assert parallel_args["features"].shape == serial_args["features"].shape
    assert np.array_equal(parallel_args["features"], serial_args["features"])

    result = optimizer.optimize(n_trials=3, n_jobs=2, results_file=None)

    assert len(result.lines) == 3


def test_mllm_train_cached_no_data(datadir, project):
    modelfile = datadir.join("mllm-model.gz")
    assert modelfile.exists()
    trainfile = datadir.join("mllm-train.gz")
    trainfile.remove()

    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm",
        config_params={"limit": 10, "language": "fi"},
        project=project,
    )

    with pytest.raises(NotInitializedException):
        mllm.train("cached")


def test_mllm_suggest_no_model(datadir, project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id="mllm", config_params={"limit": 8, "language": "fi"}, project=project
    )

    datadir.join("mllm-model.gz").remove()
    with pytest.raises(NotInitializedException):
        mllm.suggest([Document(text="example text")])

"""Unit tests for the Laya re-scoring ensemble backend in Annif"""

import importlib.util
import json
import os
import sys
import types

import pytest

import annif.backend
import annif.corpus
import annif.util
from annif.backend.laya import LayaBackend
from annif.corpus import Document
from annif.exception import (
    NotInitializedException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch


class FakeRouter:
    """Minimal stand-in for laya.Router. Answers the backend's choice
    question with a fixed probability per option. omit_probabilities
    simulates an answer that carries no probabilities at all."""

    def __init__(self, probability=0.5, fail=False, omit_probabilities=False):
        self.probability = probability
        self.fail = fail
        self.omit_probabilities = omit_probabilities
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if self.fail:
            raise RuntimeError("fake Laya failure")
        options = list(questions["the_subject"]["criteria"])
        answer = {"type": "choice", "choice": options[0]}
        if not self.omit_probabilities:
            answer["probabilities"] = {o: self.probability for o in options}
        return {"answers": {"the_subject": answer}}


def _make_backend(app_project, config_params=None, router=None):
    """Create a LayaBackend whose (expensive) Laya model is replaced
    with a fake router, so the tests do not download any checkpoints."""
    backend = LayaBackend(
        backend_id="laya",
        config_params={"sources": "dummy-en", **(config_params or {})},
        project=app_project,
    )
    backend._router = router if router is not None else FakeRouter()
    return backend


def test_laya_default_params(app_project):
    backend = _make_backend(app_project)
    expected_default_params = {
        "limit": 20,
        "question_template": "What is the text about?",
        "laya_weight": 0.25,
        "threshold": 0.0,
        "laya_model": "multilingual",
        "laya_input_chars": 0,
        "source_weights": "config",
        "source_question_template": "Is the document about {name}?",
        "source_laya_weight": 1.0,
        "source_top_k": 0,
        "laya_epochs": 1,
        "laya_loss": "soft-ce",
        "laya_micro_batch": 16,
        "laya_grad_accum": 4,
        "laya_shuffle_options": False,
        "laya_freeze_encoder": False,
        "laya_device": "auto",
    }
    actual_params = backend.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_laya_model_forwarded_to_laya(app_project):
    backend = _make_backend(
        app_project,
        config_params={"laya_model": "multilingual"},
    )
    backend.suggest([Document(text="this is some text")])

    _, _, kwargs = backend._router.calls[0]
    assert kwargs["model"] == "multilingual"


def test_laya_model_empty_restores_autorouting(app_project):
    # an empty laya_model value omits the model kwarg, so the Laya
    # Router picks the checkpoint by document language
    backend = _make_backend(
        app_project,
        config_params={"laya_model": ""},
    )
    backend.suggest([Document(text="this is some text")])

    _, _, kwargs = backend._router.calls[0]
    assert "model" not in kwargs


def test_laya_token_budgets_forwarded_to_laya(app_project):
    backend = _make_backend(
        app_project,
        config_params={"laya_max_len": 384, "laya_head_max_len": 64},
    )
    backend.suggest([Document(text="this is some text")])

    _, _, kwargs = backend._router.calls[0]
    assert kwargs["max_len"] == 384
    assert kwargs["head_max_len"] == 64


def test_laya_suggest_blends_source_and_laya_scores(app_project):
    # The dummy source suggests one subject with score 1.0; the Laya
    # fake answers option probability 0.5. The default laya_weight of
    # 0.25 blends them: 0.25 * 0.5 + 0.75 * 1.0 = 0.875.
    backend = _make_backend(app_project, router=FakeRouter(probability=0.5))
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].subject_id == app_project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == pytest.approx(0.875)


def test_laya_batch_has_one_row_per_document(app_project):
    # Regression test: the suggestion batch must contain exactly one
    # row per input document, regardless of whether documents were
    # re-scored, kept at their source scores, or had no candidates.
    backend = _make_backend(app_project)
    documents = [
        Document(text="first text"),
        Document(text="second text"),
        Document(text="third text"),
        Document(text=""),  # no candidates
    ]
    batch = backend.suggest(documents)
    assert len(batch) == len(documents)


def test_laya_question_is_choice_over_candidates(app_project):
    backend = _make_backend(app_project)
    backend.suggest([Document(text="this is some text")])

    _, questions, kwargs = backend._router.calls[0]
    # one choice question whose options are the candidate labels
    assert list(questions) == ["the_subject"]
    question = questions["the_subject"]
    assert question["type"] == "choice"
    assert question["instructions"] == "What is the text about?"
    assert list(question["criteria"]) == ["dummy"]


def test_laya_weight_averages(app_project):
    # laya_weight w: new score = w * option probability + (1 - w) * source score
    backend = _make_backend(
        app_project,
        config_params={"laya_weight": 0.7},
        router=FakeRouter(probability=0.2),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert hits[0].score == pytest.approx(0.7 * 0.2 + 0.3 * 1.0)


def test_laya_weight_one_replaces_score(app_project):
    # laya_weight 1.0 makes the option probability fully replace the
    # source score
    backend = _make_backend(
        app_project,
        config_params={"laya_weight": 1.0},
        router=FakeRouter(probability=0.5),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert hits[0].score == pytest.approx(0.5)


def test_laya_zero_probability_drops_suggestion_with_weight_one(app_project):
    # With pure replacement, an option probability of 0.0 means Laya
    # assigns the subject no likelihood and the suggestion is dropped.
    backend = _make_backend(
        app_project,
        config_params={"laya_weight": 1.0},
        router=FakeRouter(probability=0.0),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    assert len(result) == 0


def test_laya_zero_probability_survives_with_blend(app_project):
    # Averaging means one weak signal cannot fully discard a candidate:
    # probability 0.0 with source score 1.0 at the default 0.25 weight
    # -> 0.75.
    backend = _make_backend(app_project, router=FakeRouter(probability=0.0))
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.75)


def test_laya_missing_probabilities_keeps_original_scores(app_project):
    # An answer without probabilities is ignored and the candidate
    # keeps its original source score.
    backend = _make_backend(
        app_project,
        router=FakeRouter(omit_probabilities=True),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0)


def test_laya_input_chars_truncates_text(app_project):
    # Only the text sent to Laya is truncated; the sources still see
    # the full document.
    backend = _make_backend(
        app_project,
        config_params={"laya_input_chars": 4, "laya_weight": 1.0},
        router=FakeRouter(probability=0.9),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    state, _, kwargs = backend._router.calls[0]
    assert state == "this"
    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.9)


def test_laya_failure_keeps_original_scores(app_project):
    # If Laya cannot answer, the plain ensemble (merged) scores are kept.
    backend = _make_backend(app_project, router=FakeRouter(fail=True))
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_laya_threshold_filters_final_scores(app_project):
    # A project-level threshold is applied to the final blended scores
    # inside the backend, so every caller (eval, suggest, REST) honors it.
    # Blend with laya_weight 0.5: 0.5 * 0.5 + 0.5 * 1.0 = 0.75.
    backend = _make_backend(
        app_project,
        config_params={"threshold": 0.75, "laya_weight": 0.5},
        router=FakeRouter(probability=0.5),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.75)


def test_laya_threshold_drops_candidates_below_it(app_project):
    # The same 0.75 blend is dropped when the threshold is higher
    backend = _make_backend(
        app_project,
        config_params={"threshold": 0.8, "laya_weight": 0.5},
        router=FakeRouter(probability=0.5),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    assert len(result) == 0


def test_laya_is_trained_from_sources(app_project):
    backend = _make_backend(app_project)
    assert backend.is_trained


class FakeTrainConfig:
    """Stand-in for laya.train.TrainConfig: stores the knobs the
    backend passes through."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_laya_train(monkeypatch, calls):
    """Install a fake laya.train module recording finetune() calls; the
    fake writes a checkpoint layout so the backend reports trained."""

    def finetune(data, model_dir, output_dir, config=None, device=None):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump({"fine_tuned": True}, f)
        calls.append(
            {
                "data": data,
                "model_dir": model_dir,
                "output_dir": output_dir,
                "config": config,
                "device": device,
            }
        )
        return {
            "train_items": 2,
            "epoch_loss": [1.0, 0.5],
            "output_dir": output_dir,
        }

    laya_mod = types.ModuleType("laya")
    train_mod = types.ModuleType("laya.train")
    train_mod.TrainConfig = FakeTrainConfig
    train_mod.finetune = finetune
    laya_mod.train = train_mod
    monkeypatch.setitem(sys.modules, "laya", laya_mod)
    monkeypatch.setitem(sys.modules, "laya.train", train_mod)
    return finetune


def _make_corpus(tmpdir, project, lines):
    tmpfile = tmpdir.join("documents.tsv")
    tmpfile.write("".join(lines))
    return annif.corpus.DocumentFileTSV(str(tmpfile), project.subjects)


def _trained_backend(app_project, tmpdir, monkeypatch, calls, config_params=None):
    backend = _make_backend(app_project, config_params=config_params)
    backend.datadir = str(tmpdir.mkdir("data"))
    _fake_laya_train(monkeypatch, calls)
    return backend


def _read_train_rows(backend):
    path = os.path.join(backend.datadir, backend.TRAIN_FILE)
    with open(path, encoding="utf-8") as train_file:
        return [json.loads(line) for line in train_file]


def test_laya_train_writes_rows_and_finetunes(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    corpus = _make_corpus(
        tmpdir,
        app_project,
        [
            "text about dummy things\thttp://example.org/dummy\n",
            "another dummy text\thttp://example.org/dummy\n",
        ],
    )
    backend.train(corpus)

    # training data: one row per document, the same choice question the
    # backend asks at suggest time, gold probabilities keyed by option
    rows = _read_train_rows(backend)
    assert len(rows) == 2
    for row in rows:
        question = row["questions"]["the_subject"]
        assert question["type"] == "choice"
        assert question["instructions"] == "What is the text about?"
        assert list(question["criteria"]) == ["dummy"]
        assert row["gold"]["the_subject"]["probabilities"] == {"dummy": 1.0}

    # fine-tuning ran on the training data with the default knobs
    assert len(calls) == 1
    assert calls[0]["data"] == os.path.join(backend.datadir, backend.TRAIN_FILE)
    assert calls[0]["model_dir"] == "multilingual"
    assert calls[0]["output_dir"] == os.path.join(backend.datadir, backend.MODEL_DIR)
    assert calls[0]["device"] == "auto"
    assert calls[0]["config"].epochs == 1
    assert calls[0]["config"].loss == "soft-ce"
    assert calls[0]["config"].micro_batch == 16
    assert calls[0]["config"].grad_accum == 4
    assert calls[0]["config"].shuffle_options == ()
    assert calls[0]["config"].freeze_encoder is False
    assert calls[0]["config"].max_len is None
    assert calls[0]["config"].head_max_len is None

    # the fine-tuned checkpoint makes the backend trained
    assert backend.is_trained
    assert backend.modification_time is not None


def test_laya_train_custom_knobs_and_base_model(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(
        app_project,
        tmpdir,
        monkeypatch,
        calls,
        config_params={
            "laya_model": "english",
            "laya_epochs": 2,
            "laya_loss": "rlcd",
            "laya_shuffle_options": "true",
            "laya_device": "cpu",
            "laya_micro_batch": 32,
            "laya_grad_accum": 2,
            "laya_freeze_encoder": "true",
            "laya_max_len": 512,
            "laya_head_max_len": 192,
        },
    )
    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    backend.train(corpus)

    assert calls[0]["model_dir"] == "english"
    assert calls[0]["device"] == "cpu"
    assert calls[0]["config"].epochs == 2
    assert calls[0]["config"].loss == "rlcd"
    assert calls[0]["config"].shuffle_options == ("choice",)
    assert calls[0]["config"].micro_batch == 32
    assert calls[0]["config"].grad_accum == 2
    assert calls[0]["config"].freeze_encoder is True
    assert calls[0]["config"].max_len == 512
    assert calls[0]["config"].head_max_len == 192


def test_laya_train_skips_documents_without_gold_candidate(
    app_project, tmpdir, monkeypatch
):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    corpus = _make_corpus(
        tmpdir,
        app_project,
        [
            "text about dummy things\thttp://example.org/dummy\n",
            # the dummy source never suggests 'none', so this gold
            # subject is not among the question options
            "unrelated text\thttp://example.org/none\n",
        ],
    )
    backend.train(corpus)

    rows = _read_train_rows(backend)
    assert len(rows) == 1
    assert rows[0]["gold"]["the_subject"]["probabilities"] == {"dummy": 1.0}


def test_laya_train_no_gold_candidates_not_supported(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    corpus = _make_corpus(
        tmpdir, app_project, ["unrelated text\thttp://example.org/none\n"]
    )
    with pytest.raises(NotSupportedException):
        backend.train(corpus)
    assert calls == []


def test_laya_train_empty_corpus_not_supported(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    with pytest.raises(NotSupportedException):
        backend.train(annif.corpus.DocumentList([]))
    assert calls == []


def test_laya_train_cached_reuses_train_data(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    backend.train(corpus)
    assert len(calls) == 1

    backend.train("cached")

    assert len(calls) == 2
    assert calls[1]["data"] == calls[0]["data"]
    assert len(_read_train_rows(backend)) == 1


def test_laya_train_cached_without_train_data_not_initialized(
    app_project, tmpdir, monkeypatch
):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    with pytest.raises(NotInitializedException):
        backend.train("cached")
    assert calls == []


def test_laya_train_multi_source_includes_noul_questions(
    app_project, tmpdir, monkeypatch
):
    # with source_weights=laya and two sources, each training row also
    # carries the per-source noul questions with derived gold, and the
    # training-data generation does not call Laya at all (the pool is
    # the static merge)
    calls = []
    backend = _trained_backend(
        app_project,
        tmpdir,
        monkeypatch,
        calls,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
        },
    )
    corpus = _make_corpus(
        tmpdir,
        app_project,
        [
            "text about dummy things\thttp://example.org/dummy\n",
            "another dummy text\thttp://example.org/dummy\n",
        ],
    )
    backend.train(corpus)

    # no Laya inference during training-data generation
    assert backend._router.calls == []

    rows = _read_train_rows(backend)
    assert len(rows) == 2
    for row in rows:
        assert set(row["questions"]) == {"the_subject", "dummy-en", "dummy-private"}
        assert row["questions"]["the_subject"]["type"] == "choice"
        assert row["questions"]["dummy-en"] == {
            "type": "noul",
            "instructions": "Is the document about Dummy English?",
        }
        assert row["questions"]["dummy-private"] == {
            "type": "noul",
            "instructions": "Is the document about Dummy private project?",
        }
        # both dummy sources suggest the gold subject 'dummy'
        assert row["gold"]["the_subject"]["probabilities"] == {"dummy": 1.0}
        assert row["gold"]["dummy-en"]["probabilities"] == {"false": 0.0, "true": 1.0}
        assert row["gold"]["dummy-private"]["probabilities"] == {
            "false": 0.0,
            "true": 1.0,
        }


def test_laya_train_multi_source_without_laya_weights_is_choice_only(
    app_project, tmpdir, monkeypatch
):
    # with the default source_weights=config, training rows carry only
    # the choice question even with two sources
    calls = []
    backend = _trained_backend(
        app_project,
        tmpdir,
        monkeypatch,
        calls,
        config_params={"sources": "dummy-en,dummy-private"},
    )
    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    backend.train(corpus)

    rows = _read_train_rows(backend)
    assert len(rows) == 1
    assert set(rows[0]["questions"]) == {"the_subject"}
    assert set(rows[0]["gold"]) == {"the_subject"}


def test_laya_train_row_noul_gold_comes_from_source_hits(app_project):
    # unit test of _train_row: dummy-en suggests the gold subject,
    # dummy-private suggests a non-gold one -> noul gold 1.0 vs 0.0
    from annif.corpus.types import SubjectSet

    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
        },
    )
    params = backend._get_backend_params(None)
    sources = annif.util.parse_sources(params["sources"])
    noul_questions = backend._source_noul_questions(sources, params)

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    aligned_by_source = _two_source_batches(app_project)
    merged = SuggestionBatch.from_sequence(
        [[SubjectSuggestion(subject_id=dummy_id, score=1.0)]], app_project.subjects
    )
    doc = Document(text="some text", subject_set=SubjectSet([dummy_id]))

    row = backend._train_row(
        doc, 0, merged[0], sources, params, noul_questions, aligned_by_source
    )
    assert set(row["questions"]) == {"the_subject", "dummy-en", "dummy-private"}
    assert row["gold"]["the_subject"]["probabilities"] == {"dummy": 1.0}
    assert row["gold"]["dummy-en"]["probabilities"] == {"false": 0.0, "true": 1.0}
    assert row["gold"]["dummy-private"]["probabilities"] == {"false": 0.0, "true": 0.0}

    # a document whose gold is not among the candidates yields no row
    doc_none = Document(text="other text", subject_set=SubjectSet([none_id]))
    assert (
        backend._train_row(
            doc_none, 0, merged[0], sources, params, noul_questions, aligned_by_source
        )
        is None
    )


def test_laya_train_patches_laya_forward_device(app_project, tmpdir, monkeypatch):
    # _train wraps laya.train._forward so the model is moved to the batch
    # device before every forward (upstream MPS multi-question bug)
    import torch

    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)

    laya_mod = types.ModuleType("laya")
    train_mod = types.ModuleType("laya.train")
    train_mod.TrainConfig = FakeTrainConfig

    def finetune(data, model_dir, output_dir, config=None, device=None):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump({"fine_tuned": True}, f)
        calls.append({"data": data, "device": device})
        return {"train_items": 1, "epoch_loss": [0.5]}

    def _forward(model, batch, device, amp, detach_encoder):
        return model(batch, device)

    train_mod.finetune = finetune
    train_mod._forward = _forward
    laya_mod.train = train_mod
    monkeypatch.setitem(sys.modules, "laya", laya_mod)
    monkeypatch.setitem(sys.modules, "laya.train", train_mod)

    class FakeModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(1))  # cpu
            self.moved_to = None

        def parameters(self):
            return iter([self.param])

        def to(self, device):
            self.moved_to = device
            self.param = torch.nn.Parameter(torch.zeros(1, device=device))
            return self

        def __call__(self, batch, device):
            return "logits"

    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    backend.train(corpus)

    # the patch was installed on the fake laya.train
    assert getattr(train_mod, "_annif_forward_device_patched", False)
    wrapped = train_mod._forward
    assert wrapped is not _forward

    # calling the wrapped _forward with a different device moves the model first
    model = FakeModel()
    other = torch.device("meta")
    assert wrapped(model, {}, other, False, False) == "logits"
    assert model.moved_to == other
    # a second call with the model already on the device does not move again
    model.moved_to = None
    assert wrapped(model, {}, other, False, False) == "logits"
    assert model.moved_to is None


def test_laya_train_without_laya_package(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    monkeypatch.setitem(sys.modules, "laya", None)
    monkeypatch.setitem(sys.modules, "laya.train", None)
    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    with pytest.raises(OperationFailedException):
        backend.train(corpus)
    assert calls == []


def test_laya_finetuned_model_used_after_training(app_project, tmpdir, monkeypatch):
    calls = []
    backend = _trained_backend(app_project, tmpdir, monkeypatch, calls)
    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    backend.train(corpus)
    backend.suggest([Document(text="this is some text")])

    _, _, kwargs = backend._router.calls[0]
    assert kwargs["model"] == "finetuned"


def test_laya_explicit_model_param_overrides_finetuned(
    app_project, tmpdir, monkeypatch
):
    calls = []
    backend = _trained_backend(
        app_project, tmpdir, monkeypatch, calls, config_params={"laya_model": "english"}
    )
    corpus = _make_corpus(
        tmpdir, app_project, ["dummy text\thttp://example.org/dummy\n"]
    )
    backend.train(corpus)
    backend.suggest([Document(text="this is some text")])

    _, _, kwargs = backend._router.calls[0]
    assert kwargs["model"] == "english"


def test_laya_is_trained_falls_back_to_sources(app_project, tmpdir):
    backend = _make_backend(app_project)
    backend.datadir = str(tmpdir.mkdir("data"))
    # no fine-tuned checkpoint: trained state comes from the sources
    assert backend.is_trained
    assert backend._finetuned_model_dir is None


class MapRouter(FakeRouter):
    """FakeRouter variant with a per-option probability map, for tests
    that need different probabilities for different candidates."""

    def __init__(self, prob_map):
        super().__init__()
        self.prob_map = prob_map

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        options = list(questions["the_subject"]["criteria"])
        return {
            "answers": {
                "the_subject": {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {o: self.prob_map.get(o, 0.0) for o in options},
                }
            }
        }


class NoulRouter(FakeRouter):
    """FakeRouter variant that also answers the per-source noul
    questions, with a fixed P(true) per question id. A qid missing from
    noul_probs simulates an answer that lacks that question; fail_noul
    simulates the whole noul pass raising."""

    def __init__(self, noul_probs=None, probability=0.5, fail_noul=False):
        super().__init__(probability=probability)
        self.noul_probs = noul_probs or {}
        self.fail_noul = fail_noul

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if self.fail_noul and any(q["type"] == "noul" for q in questions.values()):
            raise RuntimeError("fake Laya noul failure")
        answers = {}
        for qid, question in questions.items():
            if question["type"] == "noul":
                if qid in self.noul_probs:
                    answers[qid] = {"type": "noul", "noul": self.noul_probs[qid]}
            else:
                options = list(question["criteria"])
                answers[qid] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {o: self.probability for o in options},
                }
        return {"answers": answers}


def _two_source_batches(app_project):
    """Hand-built per-source batches where dummy-en suggests 'dummy' and
    dummy-private suggests 'none', both with score 1.0."""
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    return {
        "dummy-en": SuggestionBatch.from_sequence(
            [[SubjectSuggestion(subject_id=dummy_id, score=1.0)]], app_project.subjects
        ),
        "dummy-private": SuggestionBatch.from_sequence(
            [[SubjectSuggestion(subject_id=none_id, score=1.0)]], app_project.subjects
        ),
    }


def _merge_two_sources(backend, app_project, router, config_params=None):
    """Run _merge_for_documents over the hand-built two-source batches
    and return {subject_id: score} of the single merged row."""
    backend._router = router
    params = backend._get_backend_params(None)
    sources = annif.util.parse_sources(params["sources"])
    merged = backend._merge_for_documents(
        [Document(text="some text")], _two_source_batches(app_project), sources, params
    )
    return {s.subject_id: s.score for s in merged[0]}


def test_laya_source_weights_off_by_default_sends_no_noul(app_project):
    # with the default source_weights=config no noul questions are sent,
    # only the choice question
    backend = _make_backend(
        app_project,
        config_params={"sources": "dummy-en,dummy-private"},
        router=NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1}),
    )
    backend.suggest([Document(text="this is some text")])

    assert len(backend._router.calls) == 1
    _, questions, _ = backend._router.calls[0]
    assert list(questions) == ["the_subject"]


def test_laya_source_weights_sends_one_noul_question_per_source(app_project):
    backend = _make_backend(
        app_project,
        config_params={"sources": "dummy-en,dummy-private", "source_weights": "laya"},
        router=NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1}),
    )
    backend.suggest([Document(text="this is some text")])

    # first call: the noul questions, keyed by project id, using the
    # projects' names; second call: the choice question
    assert len(backend._router.calls) == 2
    _, questions, _ = backend._router.calls[0]
    assert questions == {
        "dummy-en": {
            "type": "noul",
            "instructions": "Is the document about Dummy English?",
        },
        "dummy-private": {
            "type": "noul",
            "instructions": "Is the document about Dummy private project?",
        },
    }
    _, choice_questions, _ = backend._router.calls[1]
    assert list(choice_questions) == ["the_subject"]


def test_laya_source_question_template_placeholders(app_project):
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "source_question_template": "Topic {name} ({id})?",
        },
        router=NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1}),
    )
    backend.suggest([Document(text="this is some text")])

    _, questions, _ = backend._router.calls[0]
    assert questions["dummy-en"]["instructions"] == "Topic Dummy English (dummy-en)?"
    assert (
        questions["dummy-private"]["instructions"]
        == "Topic Dummy private project (dummy-private)?"
    )


def test_laya_per_document_weights_change_the_merge(app_project):
    # source_laya_weight=1.0: the noul scores are the weights, normalized
    # per document -> dummy 0.9, none 0.1
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "source_laya_weight": 1.0,
        },
    )
    hits = _merge_two_sources(
        backend, app_project, NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1})
    )

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    assert hits[dummy_id] == pytest.approx(0.9)
    assert hits[none_id] == pytest.approx(0.1)


def test_laya_source_laya_weight_blends_with_config_weight(app_project):
    # w = s * P_noul + (1 - s) * w_config: s=0.5, P=1.0/0.0, config
    # 0.75/0.25 -> 0.875/0.125
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en:0.75,dummy-private:0.25",
            "source_weights": "laya",
            "source_laya_weight": 0.5,
        },
    )
    hits = _merge_two_sources(
        backend, app_project, NoulRouter({"dummy-en": 1.0, "dummy-private": 0.0})
    )

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    assert hits[dummy_id] == pytest.approx(0.875)
    assert hits[none_id] == pytest.approx(0.125)


def test_laya_missing_noul_answer_uses_config_weight(app_project):
    # an answer lacking one question gives that source its config weight:
    # w = (0.8, 0.5) -> normalized 0.8/1.3 and 0.5/1.3
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "source_laya_weight": 1.0,
        },
    )
    hits = _merge_two_sources(backend, app_project, NoulRouter({"dummy-en": 0.8}))

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    assert hits[dummy_id] == pytest.approx(0.8 / 1.3)
    assert hits[none_id] == pytest.approx(0.5 / 1.3)


def test_laya_source_top_k_drops_low_weight_sources(app_project):
    # top_k=1 keeps only dummy-en; dummy-private's candidate gets weight
    # 0 and disappears from the merged row
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "source_laya_weight": 1.0,
            "source_top_k": 1,
        },
    )
    hits = _merge_two_sources(
        backend, app_project, NoulRouter({"dummy-en": 0.6, "dummy-private": 0.4})
    )

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    assert hits[dummy_id] == pytest.approx(1.0)
    assert none_id not in hits


def test_laya_source_top_k_keeps_all_when_k_covers_sources(app_project):
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "source_laya_weight": 1.0,
            "source_top_k": 2,
        },
    )
    hits = _merge_two_sources(
        backend, app_project, NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1})
    )

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    assert hits[dummy_id] == pytest.approx(0.9)
    assert hits[none_id] == pytest.approx(0.1)


def test_laya_source_weighting_failure_falls_back_to_config_weights(app_project):
    # a failed noul pass keeps the static config weights (and the top-K
    # gate does not apply to the fallback): config 0.9/0.1
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en:0.9,dummy-private:0.1",
            "source_weights": "laya",
            "source_top_k": 1,
        },
    )
    hits = _merge_two_sources(backend, app_project, NoulRouter(fail_noul=True))

    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    assert hits[dummy_id] == pytest.approx(0.9)
    assert hits[none_id] == pytest.approx(0.1)


def test_laya_weight_zero_skips_the_choice_pass(app_project):
    # laya_weight=0.0: no choice question is sent at all, the output is
    # the merged source score as-is
    backend = _make_backend(
        app_project,
        config_params={"laya_weight": 0.0},
        router=FakeRouter(probability=0.9),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    assert backend._router.calls == []
    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0)


def test_laya_moe_only_sends_noul_but_no_choice(app_project):
    # source_weights=laya + laya_weight=0.0: only the noul pass runs
    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "laya_weight": 0.0,
        },
        router=NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1}),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    assert len(backend._router.calls) == 1
    _, questions, _ = backend._router.calls[0]
    assert set(questions) == {"dummy-en", "dummy-private"}
    assert all(q["type"] == "noul" for q in questions.values())
    hits = list(result)
    assert len(hits) == 1
    assert hits[0].subject_id == app_project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == pytest.approx(1.0)


def test_laya_single_source_skips_the_noul_pass(app_project):
    # with one source there is nothing to weigh: no noul questions
    backend = _make_backend(
        app_project,
        config_params={"source_weights": "laya"},
        router=NoulRouter({"dummy-en": 0.9}),
    )
    backend.suggest([Document(text="this is some text")])

    assert len(backend._router.calls) == 1
    _, questions, _ = backend._router.calls[0]
    assert list(questions) == ["the_subject"]


def test_laya_hyperopt_tunes_source_weight_too(app_project, tmpdir):
    # with source_weights=laya the optimizer samples source_laya_weight
    # next to laya_weight and recommends two config lines
    from annif.corpus import DocumentFileTSV

    tmpfile = tmpdir.join("documents.tsv")
    tmpfile.write(
        "text about dummy things\thttp://example.org/dummy\n"
        "another dummy text\thttp://example.org/dummy\n"
        "totally unrelated content\thttp://example.org/none\n"
    )
    corpus = DocumentFileTSV(str(tmpfile), app_project.subjects)

    backend = _make_backend(
        app_project,
        config_params={
            "sources": "dummy-en,dummy-private",
            "source_weights": "laya",
            "threshold": 0.75,
        },
        router=NoulRouter({"dummy-en": 0.9, "dummy-private": 0.1}, probability=0.9),
    )
    optimizer = backend.get_hp_optimizer(corpus, "F1 score (doc avg)")

    args = optimizer._prepare(1)
    assert args["tune_sources"]
    assert len(args["noul_batches"]) == len(args["verified_batches"])

    rec = optimizer.optimize(4, 1, None)

    assert rec.score >= 0.0
    assert len(rec.lines) == 2
    assert rec.lines[0].startswith("laya_weight = ")
    assert rec.lines[1].startswith("source_laya_weight = ")
    for line in rec.lines:
        value = float(line.split("=")[1])
        assert 0.0 <= value <= 1.0


def test_laya_output_limit_ranks_by_blended_score(app_project):
    # The output must keep the top-limit candidates by BLENDED score,
    # not by the source ranking they arrived in.
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    # source ranks dummy (0.9) above none (0.1); Laya says the opposite
    backend = _make_backend(
        app_project,
        config_params={"laya_weight": 1.0, "limit": 1},
        router=MapRouter({"dummy": 0.0, "none": 1.0}),
    )
    merged = SuggestionBatch.from_sequence(
        [
            [
                SubjectSuggestion(subject_id=dummy_id, score=0.9),
                SubjectSuggestion(subject_id=none_id, score=0.1),
            ]
        ],
        app_project.subjects,
    )
    backend.initialize()
    batch = backend._rescore_with_laya(
        [Document(text="some text")], merged, backend._get_backend_params(None)
    )

    hits = list(batch[0])
    assert len(hits) == 1
    assert hits[0].subject_id == none_id
    assert hits[0].score == pytest.approx(1.0)


def test_laya_hyperopt_selects_weight(app_project, tmpdir):
    # The hyperparameter optimizer caches the (fake) Laya answers once
    # and re-combines them per trial; it optimizes only laya_weight, at
    # the configured limit and threshold, and its recommendation is a
    # single config line.
    from annif.corpus import DocumentFileTSV

    tmpfile = tmpdir.join("documents.tsv")
    tmpfile.write(
        "text about dummy things\thttp://example.org/dummy\n"
        "another dummy text\thttp://example.org/dummy\n"
        "totally unrelated content\thttp://example.org/none\n"
    )
    corpus = DocumentFileTSV(str(tmpfile), app_project.subjects)

    backend = _make_backend(
        app_project,
        config_params={"threshold": 0.75},
        router=FakeRouter(probability=0.9),
    )
    optimizer = backend.get_hp_optimizer(corpus, "F1 score (doc avg)")

    # the configured threshold is carried into the optimization
    args = optimizer._prepare(1)
    assert args["threshold"] == 0.75

    rec = optimizer.optimize(4, 1, None)

    assert rec.score >= 0.0
    assert len(rec.lines) == 1
    assert rec.lines[0].startswith("laya_weight = ")
    weight = float(rec.lines[0].split("=")[1])
    assert 0.0 <= weight <= 1.0


def test_get_backend_laya():
    if importlib.util.find_spec("laya") is None:
        pytest.skip("test requires the laya package to be installed")
    backend_type = annif.backend.get_backend("laya")
    assert backend_type is LayaBackend


@pytest.mark.skipif(
    importlib.util.find_spec("laya") is not None,
    reason="test requires that Laya is NOT installed",
)
def test_get_backend_laya_not_installed():
    with pytest.raises(ValueError) as excinfo:
        annif.backend.get_backend("laya")
    assert "Laya not available" in str(excinfo.value)

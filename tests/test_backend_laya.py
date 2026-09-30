"""Unit tests for the Laya re-scoring ensemble backend in Annif"""

import importlib.util

import pytest

import annif.backend
from annif.backend.laya import LayaBackend
from annif.corpus import Document
from annif.exception import NotSupportedException
from annif.suggestion import SuggestionBatch, SubjectSuggestion


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
        "question_template": "Which subject does this text reference?",
        "laya_weight": 0.25,
        "threshold": 0.0,
        "laya_model": "multilingual",
        "laya_input_chars": 0,
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
    assert question["instructions"] == "Which subject does this text reference?"
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


def test_laya_train_not_supported(app_project, document_corpus):
    backend = _make_backend(app_project)
    with pytest.raises(NotSupportedException):
        backend.train(document_corpus)


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

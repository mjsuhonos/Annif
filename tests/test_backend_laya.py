"""Unit tests for the Laya re-scoring ensemble backend in Annif"""

import importlib.util

import pytest

import annif.backend
from annif.backend.laya import LayaBackend
from annif.corpus import Document
from annif.exception import ConfigurationException, NotSupportedException
from annif.suggestion import SuggestionBatch, SubjectSuggestion


class FakeRouter:
    """Minimal stand-in for laya.Router. Answers the choice question
    with a fixed probability per option and every noul question with a
    fixed score. omit_probabilities and omit_noul simulate answers
    without choice probabilities / without noul scores."""

    def __init__(
        self,
        probability=0.5,
        noul=0.5,
        fail=False,
        omit_probabilities=False,
        omit_noul=False,
    ):
        self.probability = probability
        self.noul = noul
        self.fail = fail
        self.omit_probabilities = omit_probabilities
        self.omit_noul = omit_noul
        self.calls = []

    def _choice_probabilities(self, options):
        return {o: self.probability for o in options}

    def _noul_score(self, key):
        return self.noul

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if self.fail:
            raise RuntimeError("fake Laya failure")
        answers = {}
        if "the_subject" in questions:
            options = list(questions["the_subject"]["criteria"])
            answer = {"type": "choice", "choice": options[0]}
            if not self.omit_probabilities:
                answer["probabilities"] = self._choice_probabilities(options)
            answers["the_subject"] = answer
        if not self.omit_noul:
            for key, question in questions.items():
                if question["type"] == "noul":
                    answers[key] = {"type": "noul", "noul": self._noul_score(key)}
        return {"answers": answers}


class MapRouter(FakeRouter):
    """FakeRouter variant with per-option choice probabilities."""

    def __init__(self, prob_map, **kwargs):
        super().__init__(**kwargs)
        self.prob_map = prob_map

    def _choice_probabilities(self, options):
        return {o: self.prob_map.get(o, 0.0) for o in options}


class NoulRouter(FakeRouter):
    """FakeRouter variant with per-subject noul scores."""

    def __init__(self, noul_map, **kwargs):
        super().__init__(**kwargs)
        self.noul_map = noul_map

    def _noul_score(self, key):
        return self.noul_map.get(key, 0.5)


class ComboRouter(FakeRouter):
    """FakeRouter variant with per-option choice probabilities and
    per-subject noul scores, for the combined scoring mode."""

    def __init__(self, prob_map, noul_map, **kwargs):
        super().__init__(**kwargs)
        self.prob_map = prob_map
        self.noul_map = noul_map

    def _choice_probabilities(self, options):
        return {o: self.prob_map.get(o, 0.0) for o in options}

    def _noul_score(self, key):
        return self.noul_map.get(key, 0.5)


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


def _rescore_two_candidates(app_project, backend):
    """Run _rescore_with_laya on one document with two candidates:
    dummy (source score 0.9) and none (source score 0.1)."""
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
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
    return list(batch[0]), dummy_id, none_id


def test_laya_default_params(app_project):
    backend = _make_backend(app_project)
    expected_default_params = {
        "limit": 20,
        "question_template": "Which subject does this document reference?",
        "instruction": "This document is about {label}.",
        "laya_score_mode": "choice",
        "blend-alpha": 0.85,
        "blend-source": "normalized",
        "laya_model": "",
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


def test_laya_single_candidate_normalizes_to_half(app_project):
    # The dummy source suggests one subject with score 1.0. A single
    # candidate has degenerate (min == max) source and Laya score
    # distributions, so both normalize to 0.5 and the blended score is
    # 0.5 regardless of the weight and the Laya answer.
    backend = _make_backend(app_project, router=FakeRouter(probability=0.9))
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].subject_id == app_project.subjects.by_uri("http://example.org/dummy")
    assert hits[0].score == pytest.approx(0.5)


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


def test_laya_choice_mode_asks_only_the_choice_question(app_project):
    backend = _make_backend(app_project)
    backend.suggest([Document(text="this is some text")])

    _, questions, kwargs = backend._router.calls[0]
    assert list(questions) == ["the_subject"]
    question = questions["the_subject"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Which subject does this document reference?"
    assert list(question["criteria"]) == ["dummy"]


def test_laya_noul_mode_asks_only_noul_questions(app_project):
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "noul"},
    )
    backend.suggest([Document(text="this is some text")])

    _, questions, kwargs = backend._router.calls[0]
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    assert set(questions) == {str(dummy_id)}
    assert questions[str(dummy_id)]["type"] == "noul"
    assert questions[str(dummy_id)]["instructions"] == "This document is about dummy."


def test_laya_combined_mode_asks_both_question_kinds(app_project):
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "choice,noul"},
    )
    backend.suggest([Document(text="this is some text")])

    _, questions, kwargs = backend._router.calls[0]
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    assert set(questions) == {"the_subject", str(dummy_id)}
    assert questions["the_subject"]["type"] == "choice"
    assert questions[str(dummy_id)]["type"] == "noul"


def test_laya_instruction_requires_label_placeholder(app_project):
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "noul", "instruction": "no placeholder"},
    )
    with pytest.raises(ConfigurationException):
        backend.suggest([Document(text="this is some text")])


def test_laya_invalid_score_mode(app_project):
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "bogus"},
    )
    with pytest.raises(ConfigurationException):
        backend.suggest([Document(text="this is some text")])


def test_laya_invalid_blend_source(app_project):
    backend = _make_backend(
        app_project,
        config_params={"blend-source": "bogus"},
    )
    with pytest.raises(ConfigurationException):
        backend.suggest([Document(text="this is some text")])


def test_laya_raw_source_keeps_absolute_scores(app_project):
    # With blend-source=raw the source scores keep their absolute
    # values instead of being normalized per document: a single
    # candidate with source score 1.0 blends to 0.75 at blend-alpha
    # 0.5 (with the CLM-style normalization it would always be 0.5).
    backend = _make_backend(
        app_project,
        config_params={"blend-source": "raw", "blend-alpha": 0.5},
        router=FakeRouter(probability=0.9),
    )
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)


def test_laya_raw_source_blends_absolute_scores(app_project):
    # With raw source scores the blend is
    # alpha * source + (1 - alpha) * norm(laya): dummy keeps its 0.9
    # source score and none its 0.1, and the higher Laya score of none
    # lifts it above dummy.
    backend = _make_backend(
        app_project,
        config_params={"blend-source": "raw", "blend-alpha": 0.5},
        router=MapRouter({"dummy": 0.2, "none": 0.8}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [none_id, dummy_id]
    assert hits[0].score == pytest.approx(0.5 * 0.1 + 0.5 * 1.0)
    assert hits[1].score == pytest.approx(0.5 * 0.9 + 0.5 * 0.0)


def test_laya_blends_normalized_scores(app_project):
    # score = alpha * norm(source) + (1 - alpha) * norm(laya): with
    # source norms dummy=1.0 / none=0.0 and laya norms dummy=0.0 /
    # none=1.0, blend-alpha 0.75 gives dummy 0.75 and none 0.25.
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.75},
        router=MapRouter({"dummy": 0.2, "none": 0.8}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75)
    assert hits[1].score == pytest.approx(0.25)


def test_laya_blend_alpha_averages(app_project):
    # blend-alpha a: new score = a * norm(source) + (1 - a) * norm(laya)
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.3},
        router=MapRouter({"dummy": 0.2, "none": 0.8}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    # the higher noul score of none lifts it above the higher source
    # score of dummy
    assert [hit.subject_id for hit in hits] == [none_id, dummy_id]
    assert hits[0].score == pytest.approx(0.7)
    assert hits[1].score == pytest.approx(0.3)


def test_laya_alpha_zero_replaces_score(app_project):
    # blend-alpha 0.0 makes the normalized Laya score fully replace
    # the source score: dummy, the weakest Laya candidate, normalizes
    # to 0.0 and is dropped.
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.0},
        router=MapRouter({"dummy": 0.2, "none": 0.8}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [none_id]
    assert hits[0].score == pytest.approx(1.0)


def test_laya_zero_probability_drops_suggestion_with_alpha_zero(app_project):
    # With pure replacement, a Laya score of 0.0 means Laya assigns
    # the subject no likelihood: it normalizes to 0.0 and is dropped.
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.0},
        router=MapRouter({"dummy": 0.0, "none": 1.0}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [none_id]


def test_laya_zero_probability_survives_with_blend(app_project):
    # Averaging means one weak signal cannot fully discard a candidate:
    # a Laya score of 0.0 with source norm 1.0 at blend-alpha 0.75
    # gives 0.75.
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.75},
        router=MapRouter({"dummy": 0.0, "none": 1.0}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75)
    assert hits[1].score == pytest.approx(0.25)


def test_laya_missing_scores_count_as_zero(app_project):
    # A missing Laya score is NOT the source score: it counts as 0.0,
    # like in the CLM backend. A fully missing distribution is
    # degenerate and normalizes to 0.5 for every candidate.
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.75},
        router=FakeRouter(omit_probabilities=True),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75 * 1.0 + 0.25 * 0.5)
    assert hits[1].score == pytest.approx(0.25 * 0.5)


def test_laya_noul_mode_scores_candidates(app_project):
    # laya_score_mode=noul scores with the noul propositions instead
    # of the choice probabilities; the higher noul score of none lifts
    # it above the higher source score of dummy.
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "noul", "blend-alpha": 0.4},
        router=NoulRouter({str(dummy_id): 0.1, str(none_id): 0.9}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [none_id, dummy_id]
    assert hits[0].score == pytest.approx(0.6)
    assert hits[1].score == pytest.approx(0.4)


def test_laya_noul_missing_scores_count_as_zero(app_project):
    # In noul mode a missing answer also counts as 0.0, not as the
    # source score.
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "noul", "blend-alpha": 0.75},
        router=FakeRouter(omit_noul=True),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75 * 1.0 + 0.25 * 0.5)
    assert hits[1].score == pytest.approx(0.25 * 0.5)


def test_laya_combined_mode_averages_answers(app_project):
    # In the combined mode the Laya value of a candidate is the average
    # of the two answers. Here they disagree (choice prefers dummy,
    # noul prefers none), so both average to 0.5, a degenerate set, and
    # the source ranking decides.
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "choice,noul", "blend-alpha": 0.75},
        router=ComboRouter(
            {"dummy": 0.8, "none": 0.2},
            {str(dummy_id): 0.2, str(none_id): 0.8},
        ),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75 * 1.0 + 0.25 * 0.5)
    assert hits[1].score == pytest.approx(0.25 * 0.5)


def test_laya_combined_mode_uses_available_answers(app_project):
    # In the combined mode a candidate is scored with the answers that
    # exist: when the noul answers are missing entirely, scoring falls
    # back to the choice probabilities alone.
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "choice,noul", "blend-alpha": 0.75},
        router=MapRouter({"dummy": 0.2, "none": 0.8}, omit_noul=True),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75)
    assert hits[1].score == pytest.approx(0.25)


def test_laya_combined_mode_is_order_insensitive(app_project):
    # "noul,choice" is the same mode as "choice,noul"
    dummy_id = app_project.subjects.by_uri("http://example.org/dummy")
    none_id = app_project.subjects.by_uri("http://example.org/none")
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "noul,choice", "blend-alpha": 0.75},
        router=ComboRouter(
            {"dummy": 0.8, "none": 0.2},
            {str(dummy_id): 0.2, str(none_id): 0.8},
        ),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert [hit.subject_id for hit in hits] == [dummy_id, none_id]
    assert hits[0].score == pytest.approx(0.75 * 1.0 + 0.25 * 0.5)
    assert hits[1].score == pytest.approx(0.25 * 0.5)


def test_laya_input_chars_truncates_text(app_project):
    # Only the text sent to Laya is truncated; the sources still see
    # the full document.
    backend = _make_backend(
        app_project,
        config_params={"laya_input_chars": 4},
        router=FakeRouter(probability=0.9),
    )
    backend.suggest([Document(text="this is some text")])

    state, _, kwargs = backend._router.calls[0]
    assert state == "this"


def test_laya_failure_keeps_original_scores(app_project):
    # If Laya cannot answer, the plain ensemble (merged) scores are
    # kept. This is different from a per-candidate miss, which counts
    # as 0.0.
    backend = _make_backend(app_project, router=FakeRouter(fail=True))
    result = backend.suggest([Document(text="this is some text")])[0]

    hits = list(result)
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_laya_is_trained_from_sources(app_project):
    backend = _make_backend(app_project)
    assert backend.is_trained


def test_laya_train_not_supported(app_project, document_corpus):
    backend = _make_backend(app_project)
    with pytest.raises(NotSupportedException):
        backend.train(document_corpus)


def test_laya_output_limit_ranks_by_blended_score(app_project):
    # The output must keep the top-limit candidates by BLENDED score,
    # not by the source ranking they arrived in.
    backend = _make_backend(
        app_project,
        config_params={"blend-alpha": 0.0, "limit": 1},
        router=MapRouter({"dummy": 0.0, "none": 1.0}),
    )
    hits, dummy_id, none_id = _rescore_two_candidates(app_project, backend)

    assert len(hits) == 1
    assert hits[0].subject_id == none_id
    assert hits[0].score == pytest.approx(1.0)


def _hyperopt_corpus(app_project, tmpdir):
    """A tiny labeled corpus for the hyperparameter optimizer tests."""
    from annif.corpus import DocumentFileTSV

    tmpfile = tmpdir.join("documents.tsv")
    tmpfile.write(
        "text about dummy things\thttp://example.org/dummy\n"
        "another dummy text\thttp://example.org/dummy\n"
        "totally unrelated content\thttp://example.org/none\n"
    )
    return DocumentFileTSV(str(tmpfile), app_project.subjects)


def test_laya_hyperopt_selects_alpha(app_project, tmpdir):
    # With a single-mode config the caching pass asks only that mode's
    # questions, the trials tune blend-alpha and blend-source, and the
    # recommendation has one line per hyperparameter.
    corpus = _hyperopt_corpus(app_project, tmpdir)
    backend = _make_backend(
        app_project,
        router=FakeRouter(probability=0.9),
    )
    optimizer = backend.get_hp_optimizer(corpus, "F1 score (doc avg)")

    args = optimizer._prepare(1)
    assert args["modes"] == ["choice"]

    # only the choice answers were cached by the single inference pass
    first_doc = args["verified_batches"][0][0]
    assert set(first_doc[1]) == {"choice"}

    rec = optimizer.optimize(4, 1, None)

    assert rec.score >= 0.0
    assert len(rec.lines) == 2
    assert rec.lines[0].startswith("blend-alpha = ")
    alpha = float(rec.lines[0].split("=")[1])
    assert 0.0 <= alpha <= 1.0
    assert rec.lines[1] in ("blend-source = normalized", "blend-source = raw")


def test_laya_hyperopt_selects_alpha_and_mode(app_project, tmpdir):
    # With the combined mode configured the caching pass asks both
    # question kinds, the trials can compare all three modes, and the
    # recommendation has one config line per hyperparameter.
    corpus = _hyperopt_corpus(app_project, tmpdir)
    backend = _make_backend(
        app_project,
        config_params={"laya_score_mode": "choice,noul"},
        router=FakeRouter(probability=0.9),
    )
    optimizer = backend.get_hp_optimizer(corpus, "F1 score (doc avg)")

    args = optimizer._prepare(1)
    assert args["modes"] == ["choice", "noul", "choice,noul"]

    # both answer kinds were cached by the single inference pass
    first_doc = args["verified_batches"][0][0]
    assert set(first_doc[1]) == {"choice", "noul"}

    rec = optimizer.optimize(4, 1, None)

    assert rec.score >= 0.0
    assert len(rec.lines) == 3
    assert rec.lines[0].startswith("blend-alpha = ")
    alpha = float(rec.lines[0].split("=")[1])
    assert 0.0 <= alpha <= 1.0
    assert rec.lines[1] in ("blend-source = normalized", "blend-source = raw")
    assert rec.lines[2] in (
        "laya_score_mode = choice",
        "laya_score_mode = noul",
        "laya_score_mode = choice,noul",
    )


def test_laya_hyperopt_prepare_logs_progress(app_project, tmpdir):
    # The caching pass counts the documents up front and reports m/N
    # progress per document batch at INFO level. The backend's info()
    # is stubbed on the instance so the assertions do not depend on the
    # logging machinery.
    corpus = _hyperopt_corpus(app_project, tmpdir)

    backend = _make_backend(app_project)
    messages = []
    backend.info = messages.append
    optimizer = backend.get_hp_optimizer(corpus, "NDCG")

    optimizer._prepare(1)

    assert any("caching answers for 3 doc(s) (choice)" in msg for msg in messages)
    # 3 documents with 1 candidate each: 1 choice question per doc
    assert any("cached 3/3 doc(s) (100.0%)" in msg for msg in messages)
    assert any("3 question(s)" in msg for msg in messages)


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

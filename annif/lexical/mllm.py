"""MLLM (Maui-like Lexical Matchin) model for Annif"""

from __future__ import annotations

import collections
import math
from enum import IntEnum
from statistics import mean
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from rdflib.namespace import SKOS
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

import annif.parallel
import annif.util
from annif.exception import OperationFailedException
from annif.lexical.tokenset import TokenSet, TokenSetIndex
from annif.lexical.util import (
    get_subject_labels,
    make_collection_matrix,
    make_relation_matrix,
)

if TYPE_CHECKING:
    from collections import defaultdict

    from rdflib.graph import Graph
    from rdflib.term import URIRef

    from annif.analyzer import Analyzer
    from annif.corpus.document import DocumentCorpus
    from annif.vocab import AnnifVocabulary

Term = collections.namedtuple("Term", "subject_id label is_pref")

Match = collections.namedtuple("Match", "subject_id is_pref n_tokens pos ambiguity")

CandidateBlock = collections.namedtuple("CandidateBlock", "subject_ids feats")

ModelData = collections.namedtuple(
    "ModelData", "broader narrower related collection " + "doc_freq subj_freq idf"
)

Feature = IntEnum(
    "Feature",
    "freq doc_freq subj_freq tfidf is_pref n_tokens ambiguity "
    + "first_occ last_occ spread doc_length "
    + "broader narrower related collection",
    start=0,
)

# columns of the compact per-document feature block produced by
# conflate_matches, in the same order as the corresponding Feature enum
# members; the remaining Feature columns (doc_freq, subj_freq, tfidf,
# broader, narrower, related, collection) are derived from the model data
MatchFeature = IntEnum(
    "MatchFeature",
    "freq is_pref n_tokens ambiguity "
    + "first_occ last_occ spread doc_length",
    start=0,
)


def conflate_matches(matches: list[Match], doc_length: int) -> CandidateBlock:
    """Condense the matches for each subject into one row of a compact
    per-document candidate block (subject IDs + match-level features)."""
    subj_matches = collections.defaultdict(list)
    for match in matches:
        subj_matches[match.subject_id].append(match)

    subject_ids = np.zeros(len(subj_matches), dtype=np.int32)
    # float64 so that the feature values are bit-identical to the previous
    # per-candidate namedtuples (the tfidf feature is computed from the freq
    # value in float64 before the final float32 rounding)
    feats = np.zeros((len(subj_matches), len(MatchFeature)), dtype=np.float64)
    for idx, (subject_id, subj_ms) in enumerate(subj_matches.items()):
        subject_ids[idx] = subject_id
        feats[idx, MatchFeature.freq] = len(subj_ms) / doc_length
        feats[idx, MatchFeature.is_pref] = mean((float(m.is_pref) for m in subj_ms))
        feats[idx, MatchFeature.n_tokens] = mean(m.n_tokens for m in subj_ms)
        feats[idx, MatchFeature.ambiguity] = mean(m.ambiguity for m in subj_ms)
        feats[idx, MatchFeature.first_occ] = subj_ms[0].pos / doc_length
        feats[idx, MatchFeature.last_occ] = subj_ms[-1].pos / doc_length
        feats[idx, MatchFeature.spread] = (
            subj_ms[-1].pos - subj_ms[0].pos
        ) / doc_length
        feats[idx, MatchFeature.doc_length] = doc_length
    return CandidateBlock(subject_ids, feats)


def generate_candidates(
    text: str,
    analyzer: Analyzer,
    vectorizer: CountVectorizer,
    index: TokenSetIndex,
) -> CandidateBlock:
    sentences = analyzer.tokenize_sentences(text)
    sent_tokens = vectorizer.transform(sentences)
    matches = []

    for sent_idx, token_matrix in enumerate(sent_tokens):
        tset = TokenSet(token_matrix.nonzero()[1])
        for ts, ambiguity in index.search(tset):
            matches.append(
                Match(
                    subject_id=ts.subject_id,
                    is_pref=ts.is_pref,
                    n_tokens=len(ts),
                    pos=sent_idx,
                    ambiguity=ambiguity,
                )
            )

    return conflate_matches(matches, len(sentences))


def candidates_to_features(
    candidates: CandidateBlock, mdata: "ModelData"
) -> np.ndarray:
    """Convert a CandidateBlock to a NumPy feature matrix"""

    subject_ids = candidates.subject_ids
    feats = candidates.feats
    n = len(subject_ids)
    c_ids = subject_ids.tolist()

    matrix = np.zeros((n, len(Feature)), dtype=np.float32)

    # copy the match-level feature columns in one go; the MatchFeature
    # column order matches the corresponding Feature members
    for mf in MatchFeature:
        matrix[:, Feature[mf.name]] = feats[:, mf]

    # subject-specific values: look up once per unique subject, then scatter
    uniq, inv = np.unique(subject_ids, return_inverse=True)
    doc_freq = np.array([mdata.doc_freq[s] for s in uniq.tolist()])
    subj_freq = np.array([mdata.subj_freq.get(s, 1) - 1 for s in uniq.tolist()])
    idf = np.array([mdata.idf[s] for s in uniq.tolist()])
    matrix[:, Feature.doc_freq] = doc_freq[inv]
    matrix[:, Feature.subj_freq] = subj_freq[inv]
    matrix[:, Feature.tfidf] = feats[:, MatchFeature.freq] * idf[inv]

    # relation features: sparse matrix operations on a vector marking the
    # candidate subjects, then gather the rows/columns of the candidates
    c_vec = np.zeros(mdata.related.shape[0], dtype=bool)
    c_vec[c_ids] = True
    broader = np.asarray(mdata.broader.multiply(c_vec).sum(axis=1))
    narrower = np.asarray(mdata.narrower.multiply(c_vec).sum(axis=1))
    related = np.asarray(mdata.related.multiply(c_vec).sum(axis=1))
    collection = np.asarray(
        mdata.collection.multiply(c_vec).T.dot(mdata.collection).sum(axis=0)
    )
    matrix[:, Feature.broader] = broader[c_ids, 0] / n
    matrix[:, Feature.narrower] = narrower[c_ids, 0] / n
    matrix[:, Feature.related] = related[c_ids, 0] / n
    matrix[:, Feature.collection] = collection[0, c_ids] / n
    return matrix


def create_classifier(params: dict[str, Any]) -> BaggingClassifier:
    return BaggingClassifier(
        DecisionTreeClassifier(
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
        ),
        max_samples=float(params["max_samples"]),
    )


def prediction_to_list(
    scores: np.ndarray, subject_ids: np.ndarray
) -> list[tuple[np.float64, int]]:
    subj_scores = [
        (score[1], int(subj_id)) for score, subj_id in zip(scores, subject_ids)
    ]
    return sorted(subj_scores, reverse=True)


class MLLMCandidateGenerator(annif.parallel.BaseWorker):
    @classmethod
    def generate_candidates(cls, doc_subject_set, text):
        candidates = generate_candidates(text, **cls.args)  # pragma: no cover
        return doc_subject_set, candidates  # pragma: no cover


class MLLMFeatureConverter(annif.parallel.BaseWorker):
    @classmethod
    def candidates_to_features(cls, candidates):
        return candidates_to_features(candidates, **cls.args)  # pragma: no cover


class MLLMModel:
    """Maui-like Lexical Matching model"""

    def generate_candidates(self, text: str, analyzer: Analyzer) -> CandidateBlock:
        return generate_candidates(text, analyzer, self._vectorizer, self._index)

    @property
    def _model_data(self) -> ModelData:
        return ModelData(
            broader=self._broader_matrix,
            narrower=self._narrower_matrix,
            related=self._related_matrix,
            collection=self._collection_matrix,
            doc_freq=self._doc_freq,
            subj_freq=self._subj_freq,
            idf=self._idf,
        )

    def _candidates_to_features(self, candidates: CandidateBlock) -> np.ndarray:
        return candidates_to_features(candidates, self._model_data)

    @staticmethod
    def _get_label_props(params: dict[str, Any]) -> tuple[list[URIRef], list[URIRef]]:
        pref_label_props = [SKOS.prefLabel]

        if annif.util.boolean(params["use_hidden_labels"]):
            nonpref_label_props = [SKOS.altLabel, SKOS.hiddenLabel]
        else:
            nonpref_label_props = [SKOS.altLabel]

        return (pref_label_props, nonpref_label_props)

    def _prepare_terms(
        self,
        graph: Graph,
        vocab: AnnifVocabulary,
        params: dict[str, Any],
    ) -> tuple[list[Term], list[int]]:
        pref_label_props, nonpref_label_props = self._get_label_props(params)

        terms = []
        subject_ids = []
        for subj_id, subject in vocab.subjects.active:
            subject_ids.append(subj_id)

            for label in get_subject_labels(
                graph, subject.uri, pref_label_props, params["language"]
            ):
                terms.append(Term(subject_id=subj_id, label=label, is_pref=True))

            for label in get_subject_labels(
                graph, subject.uri, nonpref_label_props, params["language"]
            ):
                terms.append(Term(subject_id=subj_id, label=label, is_pref=False))

        return (terms, subject_ids)

    def _prepare_relations(self, graph: Graph, vocab: AnnifVocabulary) -> None:
        self._broader_matrix = make_relation_matrix(graph, vocab, SKOS.broader)
        self._narrower_matrix = make_relation_matrix(graph, vocab, SKOS.narrower)
        self._related_matrix = make_relation_matrix(graph, vocab, SKOS.related)
        self._collection_matrix = make_collection_matrix(graph, vocab)

    def _prepare_train_index(
        self,
        vocab: AnnifVocabulary,
        analyzer: Analyzer,
        params: dict[str, Any],
    ) -> list[int]:
        graph = vocab.as_graph()
        terms, subject_ids = self._prepare_terms(graph, vocab, params)
        self._prepare_relations(graph, vocab)

        self._vectorizer = CountVectorizer(
            binary=True, tokenizer=analyzer.tokenize_words, token_pattern=None
        )
        label_corpus = self._vectorizer.fit_transform((t.label for t in terms))

        # frequency of each token used in labels - how rare each word is
        token_freq = np.bincount(label_corpus.indices, minlength=label_corpus.shape[1])

        self._index = TokenSetIndex()
        for term, label_matrix in zip(terms, label_corpus):
            tokens = label_matrix.nonzero()[1]
            # sort tokens by frequency - use the rarest token as index key
            tokens = sorted(tokens, key=token_freq.__getitem__)
            tset = TokenSet(tokens, term.subject_id, term.is_pref)
            self._index.add(tset)

        return subject_ids

    def _prepare_train_data(
        self, corpus: DocumentCorpus, analyzer: Analyzer, n_jobs: int
    ) -> tuple[list[CandidateBlock], np.ndarray]:
        # frequency of subjects (by id) in the generated candidates
        self._doc_freq = collections.Counter()
        # frequency of manually assigned subjects ("domain keyphraseness")
        self._subj_freq = collections.Counter()
        train_x = []
        train_ys = []

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        cg_args = {
            "analyzer": analyzer,
            "vectorizer": self._vectorizer,
            "index": self._index,
        }

        with pool_class(
            jobs, initializer=MLLMCandidateGenerator.init, initargs=(cg_args,)
        ) as pool:
            params = ((doc.subject_set, doc.text) for doc in corpus.documents)
            for doc_subject_ids, candidates in pool.starmap(
                MLLMCandidateGenerator.generate_candidates, params, 10
            ):
                self._subj_freq.update(doc_subject_ids)
                self._doc_freq.update(candidates.subject_ids.tolist())
                train_x.append(candidates)
                train_ys.append(
                    np.isin(candidates.subject_ids, list(doc_subject_ids))
                )

        # concatenate the per-document boolean label arrays into one
        # (n_total_candidates,) train_y vector
        train_y = np.concatenate(train_ys) if train_ys else np.empty(0, dtype=bool)

        return (train_x, train_y)

    def _calculate_idf(
        self, subject_ids: list[int], doc_count: int
    ) -> defaultdict[int, float]:
        idf = collections.defaultdict(float)
        for subj_id in subject_ids:
            idf[subj_id] = math.log((doc_count + 1) / (self._doc_freq[subj_id] + 1)) + 1

        return idf

    def _prepare_features(
        self, train_x: list[CandidateBlock], features: np.ndarray, n_jobs: int
    ) -> None:
        """Fill the preallocated feature matrix, one document at a time"""
        fc_args = {"mdata": self._model_data}
        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        with pool_class(
            jobs, initializer=MLLMFeatureConverter.init, initargs=(fc_args,)
        ) as pool:
            start = 0
            for block_features in pool.imap(
                MLLMFeatureConverter.candidates_to_features, train_x, 10
            ):
                end = start + len(block_features)
                features[start:end] = block_features
                start = end

    def prepare_train(
        self,
        corpus: DocumentCorpus,
        vocab: AnnifVocabulary,
        analyzer: Analyzer,
        params: dict[str, Any],
        n_jobs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # create an index from the vocabulary terms
        subject_ids = self._prepare_train_index(vocab, analyzer, params)

        # convert the corpus into train data
        train_x, train_y = self._prepare_train_data(corpus, analyzer, n_jobs)

        # precalculate idf values for all candidate subjects
        self._idf = self._calculate_idf(subject_ids, len(train_x))

        # convert the train data into feature values; the total number of
        # candidates is known from train_y, so the final feature matrix can
        # be preallocated and filled in place (no list of per-document
        # matrices, no final vstack copy)
        features = np.zeros((len(train_y), len(Feature)), dtype=np.float32)
        self._prepare_features(train_x, features, n_jobs)

        return (features, train_y)

    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        params: dict[str, Any],
    ) -> None:
        # fit the model on the training corpus
        self._classifier = create_classifier(params)
        self._classifier.fit(train_x, train_y)
        # sanity check: verify that the classifier has seen both classes
        if self._classifier.n_classes_ != 2:
            raise OperationFailedException(
                "Unable to create classifier: "
                + "Not enough positive and negative examples "
                + "in the training data. Please check that your training "
                + "data matches your vocabulary."
            )

    def predict(self, candidates: CandidateBlock) -> list[tuple[np.float64, int]]:
        if not len(candidates.subject_ids):
            return []
        features = self._candidates_to_features(candidates)
        scores = self._classifier.predict_proba(features)
        return prediction_to_list(scores, candidates.subject_ids)

    def save(self, filename: str) -> list[str]:
        return joblib.dump(self, filename)

    @staticmethod
    def load(filename: str) -> MLLMModel:
        return joblib.load(filename)

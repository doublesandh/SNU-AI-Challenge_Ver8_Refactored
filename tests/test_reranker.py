from __future__ import annotations

import numpy as np
import pytest

from snuai import perm
from snuai.infer.engine import Qwen3VLRerankerEngine
from snuai.infer.scorers import PermutationReranker, RerankerPairwiseJudge
from snuai.prompting import (RERANKER_SYSTEM_TEXT, build_reranker_messages,
                             extract_media)


def _texts(messages: list[dict]) -> str:
    return "".join(
        item["text"]
        for message in messages
        for item in message["content"]
        if item["type"] == "text"
    )


def test_reranker_prompt_uses_official_query_document_contract():
    messages = build_reranker_messages("make tea", ["f2", "f0", "f3", "f1"])

    assert messages[0]["content"][0]["text"] == RERANKER_SYSTEM_TEXT
    text = _texts(messages)
    assert "<Instruct>:" in text
    assert "<Query>: Storyline: make tea" in text
    assert "<Document>:" in text
    assert "Candidate earliest:" in text
    images, videos = extract_media(messages)
    assert images == ["f2", "f0", "f3", "f1"]
    assert videos == []


class _PermutationOracle:
    def __init__(self, target_order):
        self.target_order = list(target_order)

    def relevance_score(self, messages):
        images, _ = extract_media(messages)
        return 1.0 if images == self.target_order else 0.0


def test_permutation_reranker_returns_canonical_rank_index_scores():
    target_rank = (1, 3, 0, 2)
    target_order = perm.rank_to_order(target_rank)
    images = ["f0", "f1", "f2", "f3"]
    scorer = PermutationReranker(
        _PermutationOracle([images[i] for i in target_order]))

    scores = scorer.scores("story", images)

    assert scores.shape == (24,)
    assert int(scores.argmax()) == perm.index_of(target_rank)


def test_permutation_reranker_rejects_wrong_frame_count():
    scorer = PermutationReranker(_PermutationOracle([]))
    with pytest.raises(ValueError, match="expected 4 images"):
        scorer.scores("story", ["f0", "f1"])


class _PairwiseOracle:
    def relevance_logit(self, messages):
        images, _ = extract_media(messages)
        return 3.0 if images == ["early", "late"] else -3.0


def test_reranker_pairwise_judge_compares_both_orders():
    judge = RerankerPairwiseJudge(_PairwiseOracle())
    p = judge.p_earlier("story", "early", "late")
    reverse = judge.p_earlier("story", "late", "early")

    assert p > 0.99
    assert reverse < 0.01
    assert p + reverse == pytest.approx(1.0)


@pytest.mark.parametrize("logit, expected", [(1000.0, 1.0), (-1000.0, 0.0), (0.0, 0.5)])
def test_relevance_score_sigmoid_is_numerically_stable(logit, expected):
    engine = object.__new__(Qwen3VLRerankerEngine)
    engine.relevance_logit = lambda _messages: logit
    assert engine.relevance_score([]) == pytest.approx(expected)
    assert np.isfinite(engine.relevance_score([]))

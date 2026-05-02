"""
Unit tests for ScoreProcessor.

ScoreProcessor parses the OpenRouter/Cohere rerank API response into
scored result dicts. No external calls — pure logic tests.

Return format: list[dict] with keys: text, score
Sorted by score DESC. min_score threshold with min-1 guarantee.
"""
from __future__ import annotations

import pytest
from src.clients.reranker.score_processor import ScoreProcessor


def make_response(results: list[dict]) -> dict:
    return {"results": results}


DOCUMENTS = ["Doc A about BPJS", "Doc B about pajak", "Doc C unrelated"]


# ── Basic parsing ──────────────────────────────────────────────────────────────

def test_returns_list_of_dicts():
    resp = make_response([
        {"index": 0, "relevance_score": 0.9, "document": {"text": DOCUMENTS[0]}},
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=5)
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "text" in result[0]
    assert "score" in result[0]


def test_sorted_by_score_desc():
    resp = make_response([
        {"index": 2, "relevance_score": 0.3, "document": {"text": DOCUMENTS[2]}},
        {"index": 0, "relevance_score": 0.9, "document": {"text": DOCUMENTS[0]}},
        {"index": 1, "relevance_score": 0.6, "document": {"text": DOCUMENTS[1]}},
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=5)
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results():
    resp = make_response([
        {"index": 0, "relevance_score": 0.9, "document": {"text": DOCUMENTS[0]}},
        {"index": 1, "relevance_score": 0.6, "document": {"text": DOCUMENTS[1]}},
        {"index": 2, "relevance_score": 0.3, "document": {"text": DOCUMENTS[2]}},
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=2)
    assert len(result) == 2


def test_empty_results_returns_empty_list():
    result = ScoreProcessor.process(make_response([]), DOCUMENTS, top_k=5)
    assert result == []


# ── min_score threshold ────────────────────────────────────────────────────────

def test_min_score_filters_low_relevance():
    resp = make_response([
        {"index": 0, "relevance_score": 0.9, "document": {"text": DOCUMENTS[0]}},
        {"index": 1, "relevance_score": 0.2, "document": {"text": DOCUMENTS[1]}},
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=5, min_score=0.5)
    assert all(r["score"] >= 0.5 for r in result)
    assert len(result) == 1


def test_min_score_guarantees_at_least_one_result():
    """Even if all scores below threshold, must return 1 item (highest scored)."""
    resp = make_response([
        {"index": 0, "relevance_score": 0.1, "document": {"text": DOCUMENTS[0]}},
        {"index": 1, "relevance_score": 0.05, "document": {"text": DOCUMENTS[1]}},
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=5, min_score=0.8)
    assert len(result) == 1
    assert result[0]["score"] == 0.1  # highest of the below-threshold items


def test_no_min_score_returns_all():
    resp = make_response([
        {"index": 0, "relevance_score": 0.9, "document": {"text": DOCUMENTS[0]}},
        {"index": 1, "relevance_score": 0.1, "document": {"text": DOCUMENTS[1]}},
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=5, min_score=0.0)
    assert len(result) == 2


# ── Fallback to original documents list ───────────────────────────────────────

def test_falls_back_to_document_list_when_no_document_key():
    resp = make_response([
        {"index": 0, "relevance_score": 0.85},  # no "document" key
    ])
    result = ScoreProcessor.process(resp, DOCUMENTS, top_k=5)
    assert result[0]["text"] == DOCUMENTS[0]
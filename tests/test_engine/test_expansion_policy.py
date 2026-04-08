"""Tests for expansion policy routing."""
import pytest

from lossless_agent.engine.expansion_policy import (
    ExpansionPolicy,
    PolicyAction,
    PolicyDecision,
)
from lossless_agent.store.models import Summary


def _make_summary(summary_id="s1", depth=0, token_count=100, content="test content"):
    return Summary(
        summary_id=summary_id,
        conversation_id=1,
        kind="leaf",
        depth=depth,
        content=content,
        token_count=token_count,
        source_token_count=200,
        earliest_at="2025-01-01",
        latest_at="2025-01-02",
        model="test",
        created_at="2025-01-02",
    )


@pytest.fixture
def policy():
    return ExpansionPolicy()


class TestDecideNoCandidates:
    def test_no_candidates_answer_directly(self, policy):
        decision = policy.decide("what happened?", [], token_budget=4000)
        assert decision.action == PolicyAction.ANSWER_DIRECTLY
        assert any("nothing" in r.lower() or "no" in r.lower() for r in decision.reasons)


class TestDecideRecursionLimit:
    def test_at_max_depth_answer_directly(self, policy):
        candidates = [_make_summary()]
        decision = policy.decide("what?", candidates, token_budget=4000, current_depth=1)
        assert decision.action == PolicyAction.ANSWER_DIRECTLY
        assert any("recursion" in r.lower() or "depth" in r.lower() for r in decision.reasons)


class TestDecideLowRisk:
    def test_low_token_risk_expand_shallow(self, policy):
        # 1 candidate, 500 tokens estimated, budget=4000 -> ratio ~0.125 (low)
        candidates = [_make_summary()]
        decision = policy.decide("tell me about X", candidates, token_budget=4000)
        assert decision.action == PolicyAction.EXPAND_SHALLOW


class TestDecideModerateRisk:
    def test_moderate_risk_few_candidates_expand_shallow(self, policy):
        # 3 candidates, 500*3=1500, budget=3000 -> ratio=0.5 (moderate), candidates<=3
        candidates = [_make_summary(f"s{i}") for i in range(3)]
        decision = policy.decide("tell me about X", candidates, token_budget=3000)
        assert decision.action == PolicyAction.EXPAND_SHALLOW

    def test_moderate_risk_many_candidates_delegate(self, policy):
        # 4 candidates, 500*4=2000, budget=4000 -> ratio=0.5 (moderate), candidates>3
        candidates = [_make_summary(f"s{i}") for i in range(4)]
        decision = policy.decide("tell me about X", candidates, token_budget=4000)
        assert decision.action == PolicyAction.DELEGATE_TRAVERSAL


class TestDecideHighRisk:
    def test_high_risk_delegates(self, policy):
        # 6 candidates, 500*6=3000, budget=3000 -> ratio=1.0 (high)
        candidates = [_make_summary(f"s{i}") for i in range(6)]
        decision = policy.decide("tell me about X", candidates, token_budget=3000)
        assert decision.action == PolicyAction.DELEGATE_TRAVERSAL


class TestBroadTimeRange:
    def test_detects_last_month(self, policy):
        assert policy._detect_broad_time_range("what happened last month") is True

    def test_detects_past_year(self, policy):
        assert policy._detect_broad_time_range("over the past year") is True

    def test_detects_history(self, policy):
        assert policy._detect_broad_time_range("show me all history") is True

    def test_no_time_range(self, policy):
        assert policy._detect_broad_time_range("what is X?") is False

    def test_broad_time_increases_estimate(self, policy):
        # 2 candidates base: 500*2=1000, budget=4000 -> 0.25 (low, EXPAND_SHALLOW)
        # With time multiplier 1.5: 1500/4000=0.375 (moderate, but <=3 -> EXPAND_SHALLOW still)
        # Let's pick numbers where it tips: 3 candidates, budget=3000
        # base: 1500/3000=0.5 moderate, <=3 -> EXPAND_SHALLOW
        # With 1.5x: 2250/3000=0.75 -> high -> DELEGATE
        candidates = [_make_summary(f"s{i}") for i in range(3)]
        decision = policy.decide("what happened over the past year", candidates, token_budget=3000)
        assert decision.action == PolicyAction.DELEGATE_TRAVERSAL


class TestMultiHop:
    def test_detects_compare(self, policy):
        candidates = [_make_summary()]
        assert policy._detect_multi_hop("compare A and B", candidates, 0) is True

    def test_detects_relationship(self, policy):
        candidates = [_make_summary()]
        assert policy._detect_multi_hop("relationship between X and Y", candidates, 0) is True

    def test_detects_high_depth(self, policy):
        candidates = [_make_summary()]
        assert policy._detect_multi_hop("simple query", candidates, 3) is True

    def test_detects_many_candidates(self, policy):
        candidates = [_make_summary(f"s{i}") for i in range(5)]
        assert policy._detect_multi_hop("simple query", candidates, 0) is True

    def test_no_multi_hop(self, policy):
        candidates = [_make_summary()]
        assert policy._detect_multi_hop("what is X", candidates, 0) is False


class TestPolicyDecisionDataclass:
    def test_decision_has_reasons(self, policy):
        decision = policy.decide("query", [], token_budget=4000)
        assert isinstance(decision, PolicyDecision)
        assert isinstance(decision.reasons, list)
        assert len(decision.reasons) > 0

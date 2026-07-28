"""The summariser must instruct a SUMMARY, not a conversation continuation (regression for the bug
where prompts contained only a token target, so the model continued the chat instead)."""
from lossless_agent.engine.summarize_prompt import build_leaf_prompt, build_condensed_prompt


def test_leaf_prompt_instructs_summary_not_continuation():
    p = build_leaf_prompt("[user] hi\n[assistant] fixed the bug in foo.py", 300)
    low = p.lower()
    assert "summary" in low
    assert "not a reply" in low            # explicit anti-continuation instruction
    assert "do not continue" in low
    assert "<messages>" in p and "300" in p


def test_condensed_prompt_instructs_summary():
    p = build_condensed_prompt("leaf1\nleaf2", 200, depth=1)
    low = p.lower()
    assert "summary" in low and "not a reply" in low
    assert "<summaries>" in p and "200" in p

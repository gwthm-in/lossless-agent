"""Tests for SessionPatternMatcher."""
import pytest

from lossless_agent.engine.session_patterns import SessionPatternMatcher


class TestGlobToRegex:
    """Test the glob-to-regex conversion."""

    def test_single_star_matches_non_colon(self):
        m = SessionPatternMatcher(ignore_patterns=["user:*"])
        assert m.is_ignored("user:alice")
        assert m.is_ignored("user:bob")
        assert not m.is_ignored("user:alice:chat")

    def test_double_star_matches_any(self):
        m = SessionPatternMatcher(ignore_patterns=["user:**"])
        assert m.is_ignored("user:alice")
        assert m.is_ignored("user:alice:chat:123")

    def test_literal_match(self):
        m = SessionPatternMatcher(ignore_patterns=["debug:temp"])
        assert m.is_ignored("debug:temp")
        assert not m.is_ignored("debug:temp2")

    def test_star_in_middle(self):
        m = SessionPatternMatcher(ignore_patterns=["org:*:temp"])
        assert m.is_ignored("org:acme:temp")
        assert not m.is_ignored("org:acme:sub:temp")

    def test_double_star_in_middle(self):
        m = SessionPatternMatcher(ignore_patterns=["org:**:temp"])
        assert m.is_ignored("org:acme:temp")
        assert m.is_ignored("org:acme:sub:temp")

    def test_no_partial_match(self):
        """Pattern must match the entire session key."""
        m = SessionPatternMatcher(ignore_patterns=["user:*"])
        assert not m.is_ignored("prefix:user:alice")


class TestIsIgnored:
    def test_empty_patterns(self):
        m = SessionPatternMatcher(ignore_patterns=[])
        assert not m.is_ignored("anything")

    def test_none_patterns(self):
        m = SessionPatternMatcher()
        assert not m.is_ignored("anything")

    def test_multiple_patterns(self):
        m = SessionPatternMatcher(ignore_patterns=["debug:*", "test:**"])
        assert m.is_ignored("debug:foo")
        assert m.is_ignored("test:a:b:c")
        assert not m.is_ignored("prod:main")


class TestIsStateless:
    def test_stateless_match(self):
        m = SessionPatternMatcher(stateless_patterns=["readonly:*"])
        assert m.is_stateless("readonly:viewer")
        assert not m.is_stateless("readwrite:editor")

    def test_empty_stateless(self):
        m = SessionPatternMatcher()
        assert not m.is_stateless("anything")

    def test_multiple_stateless_patterns(self):
        m = SessionPatternMatcher(stateless_patterns=["view:*", "guest:**"])
        assert m.is_stateless("view:page")
        assert m.is_stateless("guest:user:session")
        assert not m.is_stateless("admin:panel")


class TestShouldPersist:
    def test_normal_key_persists(self):
        m = SessionPatternMatcher(
            ignore_patterns=["debug:*"],
            stateless_patterns=["view:*"],
        )
        assert m.should_persist("prod:main")

    def test_ignored_does_not_persist(self):
        m = SessionPatternMatcher(ignore_patterns=["debug:*"])
        assert not m.should_persist("debug:test")

    def test_stateless_does_not_persist(self):
        m = SessionPatternMatcher(stateless_patterns=["view:*"])
        assert not m.should_persist("view:page")

    def test_both_ignored_and_stateless(self):
        m = SessionPatternMatcher(
            ignore_patterns=["debug:*"],
            stateless_patterns=["debug:*"],
        )
        assert not m.should_persist("debug:test")


class TestEdgeCases:
    def test_empty_session_key(self):
        m = SessionPatternMatcher(ignore_patterns=["*"])
        assert m.is_ignored("")

    def test_special_regex_chars_escaped(self):
        m = SessionPatternMatcher(ignore_patterns=["user.name:*"])
        assert m.is_ignored("user.name:test")
        assert not m.is_ignored("username:test")

    def test_pattern_with_only_stars(self):
        m = SessionPatternMatcher(ignore_patterns=["**"])
        assert m.is_ignored("anything:at:all")
        assert m.is_ignored("")

"""Tests for StructuredTextExtractor."""
from __future__ import annotations

import json

import pytest

from lossless_agent.engine.structured_text import StructuredTextExtractor


class TestStructuredTextExtractor:
    def setup_method(self):
        self.extractor = StructuredTextExtractor()

    # --- Non-JSON passthrough ---

    def test_plain_text_passthrough(self):
        assert self.extractor.extract("hello world") == "hello world"

    def test_empty_string(self):
        assert self.extractor.extract("") == ""

    def test_invalid_json(self):
        assert self.extractor.extract("{not json}") == "{not json}"

    # --- Direct text fields ---

    def test_text_field(self):
        data = json.dumps({"text": "hello"})
        assert self.extractor.extract(data) == "hello"

    def test_transcript_field(self):
        data = json.dumps({"transcript": "some transcript"})
        assert self.extractor.extract(data) == "some transcript"

    def test_message_field(self):
        data = json.dumps({"message": "a message"})
        assert self.extractor.extract(data) == "a message"

    def test_summary_field(self):
        data = json.dumps({"summary": "a summary"})
        assert self.extractor.extract(data) == "a summary"

    def test_multiple_text_fields(self):
        data = json.dumps({"text": "line1", "summary": "line2"})
        result = self.extractor.extract(data)
        assert "line1" in result
        assert "line2" in result

    # --- Array fields ---

    def test_segments_array(self):
        data = json.dumps({
            "segments": [
                {"text": "segment 1"},
                {"text": "segment 2"},
            ]
        })
        result = self.extractor.extract(data)
        assert "segment 1" in result
        assert "segment 2" in result

    def test_utterances_array(self):
        data = json.dumps({
            "utterances": [
                {"transcript": "hello"},
                {"transcript": "world"},
            ]
        })
        result = self.extractor.extract(data)
        assert "hello" in result
        assert "world" in result

    def test_string_items_in_array(self):
        data = json.dumps({
            "results": ["one", "two", "three"]
        })
        result = self.extractor.extract(data)
        assert "one" in result
        assert "three" in result

    # --- Nested fields ---

    def test_data_wrapper(self):
        data = json.dumps({
            "data": {"text": "nested text"}
        })
        assert self.extractor.extract(data) == "nested text"

    def test_payload_wrapper(self):
        data = json.dumps({
            "payload": {"message": "deep message"}
        })
        assert self.extractor.extract(data) == "deep message"

    def test_content_wrapper_string(self):
        data = json.dumps({
            "content": "direct string"
        })
        assert self.extractor.extract(data) == "direct string"

    # --- Deep nesting ---

    def test_deeply_nested(self):
        data = json.dumps({
            "data": {
                "output": {
                    "result": {
                        "payload": {
                            "text": "found it"
                        }
                    }
                }
            }
        })
        assert self.extractor.extract(data) == "found it"

    def test_depth_limit(self):
        # Build very deep structure
        inner = {"text": "should not reach"}
        for _ in range(10):
            inner = {"data": inner}
        data = json.dumps(inner)
        # With max_depth=2, shouldn't reach the inner text
        result = self.extractor.extract(data, max_depth=2)
        assert result != "should not reach"

    # --- List at top level ---

    def test_top_level_list(self):
        data = json.dumps([
            {"text": "first"},
            {"text": "second"},
        ])
        result = self.extractor.extract(data)
        assert "first" in result
        assert "second" in result

    def test_top_level_string_list(self):
        data = json.dumps(["alpha", "beta"])
        result = self.extractor.extract(data)
        assert "alpha" in result
        assert "beta" in result

    # --- Scalar JSON ---

    def test_json_number(self):
        result = self.extractor.extract("42")
        assert result == "42"

    def test_json_null(self):
        result = self.extractor.extract("null")
        assert result == "null"

    # --- Whisper-like response ---

    def test_whisper_response(self):
        data = json.dumps({
            "text": "full transcript here",
            "segments": [
                {"text": "segment one", "start": 0.0, "end": 1.0},
                {"text": "segment two", "start": 1.0, "end": 2.0},
            ]
        })
        result = self.extractor.extract(data)
        assert "full transcript here" in result

    # --- Empty nested values ---

    def test_empty_text_field_skipped(self):
        data = json.dumps({"text": "", "summary": "fallback"})
        assert self.extractor.extract(data) == "fallback"

    def test_whitespace_text_field_skipped(self):
        data = json.dumps({"text": "   ", "summary": "fallback"})
        assert self.extractor.extract(data) == "fallback"

    # --- MAX_DEPTH class attribute ---

    def test_max_depth_class_attr(self):
        assert StructuredTextExtractor.MAX_DEPTH == 6

    def test_text_fields_class_attr(self):
        assert "text" in StructuredTextExtractor.TEXT_FIELDS
        assert "transcript" in StructuredTextExtractor.TEXT_FIELDS

    def test_array_fields_class_attr(self):
        assert "segments" in StructuredTextExtractor.ARRAY_FIELDS
        assert "utterances" in StructuredTextExtractor.ARRAY_FIELDS

    def test_nested_fields_class_attr(self):
        assert "content" in StructuredTextExtractor.NESTED_FIELDS
        assert "data" in StructuredTextExtractor.NESTED_FIELDS

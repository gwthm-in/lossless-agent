"""Tests for MediaAnnotator — media detection, stripping, and annotation."""

import pytest

from lossless_agent.engine.media import MediaAnnotator


@pytest.fixture
def annotator():
    return MediaAnnotator()


class TestDetectMedia:
    """RED: detect_media should identify various media patterns."""

    def test_detects_base64_data_url_image(self, annotator):
        content = "Here is an image: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        assert annotator.detect_media(content) is True

    def test_detects_base64_data_url_audio(self, annotator):
        content = "Audio: data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQA"
        assert annotator.detect_media(content) is True

    def test_detects_media_path(self, annotator):
        content = "Check this out: MEDIA:/tmp/screenshot.png"
        assert annotator.detect_media(content) is True

    def test_detects_binary_hex_payload(self, annotator):
        hex_payload = " ".join(["af" * 2] * 40)  # Long hex-like block
        assert annotator.detect_media(hex_payload) is True

    def test_detects_high_byte_binary(self, annotator):
        # 20 bytes of high-byte content
        content = "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93"
        assert annotator.detect_media(content) is True

    def test_no_media_in_plain_text(self, annotator):
        content = "This is just a normal conversation about code review."
        assert annotator.detect_media(content) is False

    def test_no_media_in_empty_string(self, annotator):
        assert annotator.detect_media("") is False


class TestStripBinaryPayloads:
    """RED: strip_binary_payloads should remove media data, keep text."""

    def test_strips_base64_data_url(self, annotator):
        content = "Before data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ After"
        result = annotator.strip_binary_payloads(content)
        assert "data:image" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_media_path(self, annotator):
        content = "Screenshot: MEDIA:/tmp/img.png taken today"
        result = annotator.strip_binary_payloads(content)
        assert "MEDIA:" not in result
        assert "Screenshot:" in result
        assert "taken today" in result

    def test_strips_hex_payloads(self, annotator):
        hex_block = " ".join(["ab"] * 40)
        content = f"Header {hex_block} Footer"
        result = annotator.strip_binary_payloads(content)
        assert len(result) < len(content)

    def test_collapses_excess_newlines(self, annotator):
        content = "Line1\n\n\n\n\ndata:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ\n\n\n\n\nLine2"
        result = annotator.strip_binary_payloads(content)
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self, annotator):
        content = "Normal text with no media."
        assert annotator.strip_binary_payloads(content) == content


class TestAnnotate:
    """RED: annotate should transform content appropriately for summarization."""

    def test_media_only_becomes_attachment_tag(self, annotator):
        # Content that is ONLY a data URL — after stripping, nothing remains
        content = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVQI12Ng"
        result = annotator.annotate(content)
        assert result == "[Media attachment]"

    def test_text_plus_media_gets_annotation(self, annotator):
        content = "Here is the diagram data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        result = annotator.annotate(content)
        assert "[with media attachment]" in result
        assert "Here is the diagram" in result
        assert "data:image" not in result

    def test_plain_text_unchanged(self, annotator):
        content = "Just discussing the architecture."
        assert annotator.annotate(content) == content

    def test_media_path_only(self, annotator):
        content = "MEDIA:/var/data/recording.ogg"
        result = annotator.annotate(content)
        assert result == "[Media attachment]"

    def test_text_with_media_path(self, annotator):
        content = "Voice memo follows MEDIA:/tmp/memo.ogg please review"
        result = annotator.annotate(content)
        assert "[with media attachment]" in result
        assert "Voice memo follows" in result
        assert "MEDIA:" not in result

    def test_multiple_media_items(self, annotator):
        content = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ and MEDIA:/tmp/other.png"
        result = annotator.annotate(content)
        # Should strip both and annotate
        assert "data:image" not in result
        assert "MEDIA:" not in result

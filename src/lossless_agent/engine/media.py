"""Media annotation: detect and strip media content before summarisation."""
from __future__ import annotations

import re


# Base64 data URLs: data:image/png;base64,... or data:audio/wav;base64,...
_DATA_URL_RE = re.compile(
    r"data:[a-zA-Z0-9]+/[a-zA-Z0-9.+\-]+;base64,[A-Za-z0-9+/=\n]{20,}",
    re.DOTALL,
)

# MEDIA:/path/to/file references
_MEDIA_PATH_RE = re.compile(r"MEDIA:/[^\s]+")

# Binary-looking payloads: long runs of non-ASCII or hex-like blocks
_BINARY_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}\s*){32,}")
_BINARY_HIGH_RE = re.compile(r"[\x80-\xff]{16,}")


class MediaAnnotator:
    """Detect and annotate media content before summarisation."""

    def detect_media(self, content: str) -> bool:
        """Return True if content contains media data."""
        if _DATA_URL_RE.search(content):
            return True
        if _MEDIA_PATH_RE.search(content):
            return True
        if _BINARY_HEX_RE.search(content):
            return True
        if _BINARY_HIGH_RE.search(content):
            return True
        return False

    def strip_binary_payloads(self, content: str) -> str:
        """Remove binary-looking payloads from content."""
        result = _DATA_URL_RE.sub("", content)
        result = _MEDIA_PATH_RE.sub("", result)
        result = _BINARY_HEX_RE.sub("", result)
        result = _BINARY_HIGH_RE.sub("", result)
        # Collapse whitespace left behind
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def annotate(self, content: str) -> str:
        """Annotate content: strip media, tag appropriately.

        - Media-only messages -> '[Media attachment]'
        - Text+media -> text + ' [with media attachment]'
        - No media -> content unchanged
        """
        if not self.detect_media(content):
            return content

        stripped = self.strip_binary_payloads(content)
        if not stripped:
            return "[Media attachment]"
        return stripped + " [with media attachment]"

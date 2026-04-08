"""Extract plain text from structured (JSON) content with recursive descent."""
from __future__ import annotations

import json
from typing import List


class StructuredTextExtractor:
    """Recursively extract text from nested JSON structures.

    Follows a priority-based field traversal strategy to find text content
    in deeply nested API responses, transcripts, and other structured data.

    MAX_DEPTH limits recursion to prevent stack overflows on adversarial input.
    """

    MAX_DEPTH: int = 6

    # Priority 1: fields that directly contain text
    TEXT_FIELDS: List[str] = [
        "text", "transcript", "transcription", "message", "summary",
    ]

    # Priority 2: array fields that contain items with text
    ARRAY_FIELDS: List[str] = [
        "segments", "utterances", "paragraphs", "alternatives",
        "words", "items", "results",
    ]

    # Priority 3: wrapper/envelope fields to recurse into
    NESTED_FIELDS: List[str] = [
        "content", "output", "result", "payload", "data", "value",
    ]

    def extract(self, content: str, max_depth: int = 6) -> str:
        """Extract text from *content*.

        If *content* is valid JSON (dict or list), recursively extract
        text fields.  Otherwise return *content* as-is.
        """
        if not content:
            return content

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

        if isinstance(parsed, dict):
            return self._extract_from_dict(parsed, max_depth)
        if isinstance(parsed, list):
            return self._extract_from_list(parsed, max_depth)

        # Scalar JSON value (string, number, bool, null)
        return str(parsed) if parsed is not None else content

    def _extract_from_dict(self, d: dict, depth: int) -> str:
        """Extract text from a dict, following field priority."""
        if depth <= 0:
            return ""

        parts: List[str] = []

        # Priority 1: direct text fields
        for field in self.TEXT_FIELDS:
            if field in d:
                val = d[field]
                if isinstance(val, str) and val.strip():
                    parts.append(val)
                elif isinstance(val, dict):
                    extracted = self._extract_from_dict(val, depth - 1)
                    if extracted:
                        parts.append(extracted)
                elif isinstance(val, list):
                    extracted = self._extract_from_list(val, depth - 1)
                    if extracted:
                        parts.append(extracted)

        if parts:
            return "\n".join(parts)

        # Priority 2: array fields
        for field in self.ARRAY_FIELDS:
            if field in d and isinstance(d[field], list):
                extracted = self._extract_from_list(d[field], depth - 1)
                if extracted:
                    parts.append(extracted)

        if parts:
            return "\n".join(parts)

        # Priority 3: nested wrapper fields
        for field in self.NESTED_FIELDS:
            if field in d:
                val = d[field]
                if isinstance(val, str) and val.strip():
                    parts.append(val)
                elif isinstance(val, dict):
                    extracted = self._extract_from_dict(val, depth - 1)
                    if extracted:
                        parts.append(extracted)
                elif isinstance(val, list):
                    extracted = self._extract_from_list(val, depth - 1)
                    if extracted:
                        parts.append(extracted)

        if parts:
            return "\n".join(parts)

        # Fallback: try all string values
        for val in d.values():
            if isinstance(val, str) and val.strip():
                parts.append(val)

        return "\n".join(parts)

    def _extract_from_list(self, items: list, depth: int) -> str:
        """Extract text from a list of items."""
        if depth <= 0:
            return ""

        parts: List[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                parts.append(item)
            elif isinstance(item, dict):
                extracted = self._extract_from_dict(item, depth - 1)
                if extracted:
                    parts.append(extracted)
            elif isinstance(item, list):
                extracted = self._extract_from_list(item, depth - 1)
                if extracted:
                    parts.append(extracted)

        return "\n".join(parts)

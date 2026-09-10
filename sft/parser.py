"""Robust parser for the raw SFT exports in data/raw/.

Contract
--------
Input:  the raw *bytes* of a tab-separated export.
Output: ``(records, report)`` where ``records`` is a list of dicts keyed by the
        header's column names, and ``report`` is a :class:`ParseReport` that
        accounts for every data line that did not become a record.

Format rules (see EXERCISE.md, Part 1):
  * encoding: UTF-8 with or without a BOM; otherwise fall back to cp1252, then
    latin-1 (which never fails). A fallback is recorded in the report.
  * line endings: ``\\n``, ``\\r\\n`` or ``\\r`` - any mix.
  * blank lines and lines starting with ``#`` are ignored.
  * the first remaining line is the header; column order comes from the header.
    Required columns: ``id``, ``instruction``, ``output``. Missing optional
    columns (``input``, ``source``) are filled with ``""``.
  * fields are tab-separated. A field that *starts* with ``"`` is quoted: it may
    contain tabs and newlines, and ``""`` inside it is a literal ``"``.
  * a quoted field that is still open after ``MAX_CONTINUATION_LINES`` extra
    physical lines is an *unterminated quote*: that one line is rejected and
    parsing resumes on the next physical line (it must not swallow the file).
  * cells are Unicode NFC-normalised, stripped of zero-width characters and
    surrounding whitespace.
  * rows with the wrong number of fields, or an empty required field, are
    rejected with a reason - never silently dropped.

Invariant: ``report.ok + len(report.rejected) == report.total``.
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED = ("id", "instruction", "output")
OPTIONAL = ("input", "source")
MAX_CONTINUATION_LINES = 8
MAX_FIELD_CHARS = 10_000
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass
class ParseReport:
    total: int = 0                   # data rows seen (header, blanks, comments excluded)
    ok: int = 0
    blank: int = 0
    comments: int = 0
    rejected: list[tuple[int, str, str]] = field(default_factory=list)   # (lineno, reason, raw)
    encoding: str = "utf-8"
    encoding_fallback: bool = False

    def summary(self) -> str:
        reasons: dict[str, int] = {}
        for _, reason, _ in self.rejected:
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        return (
            f"{self.ok}/{self.total} rows ok, {len(self.rejected)} rejected {reasons}, "
            f"encoding={self.encoding}{' (FALLBACK)' if self.encoding_fallback else ''}"
        )


def decode_bytes(raw: bytes) -> tuple[str, str, bool]:
    """Return (text, encoding_used, used_fallback)."""
    try:
        return raw.decode("utf-8-sig"), "utf-8", False     # utf-8-sig strips a leading BOM
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252", True
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1", True       # never raises


def clean_cell(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_ZERO_WIDTH).strip()


def _quote_state(s: str, delimiter: str) -> str:
    """Scan `s` with RFC-4180 rules. Returns "closed", "open" (ends inside a
    quoted field) or "malformed" (a closing quote followed by something other
    than a delimiter / line break - the usual sign that an earlier quote was
    never terminated and has swallowed the following lines)."""
    in_quotes, at_field_start, i, n = False, True, 0, len(s)
    while i < n:
        c = s[i]
        if in_quotes:
            if c == '"':
                if i + 1 < n and s[i + 1] == '"':
                    i += 2                      # escaped quote
                    continue
                in_quotes = False
                if i + 1 < n and s[i + 1] not in (delimiter, "\n"):
                    return "malformed"
        elif c == '"' and at_field_start:
            in_quotes = True
        at_field_start = (not in_quotes) and c == delimiter
        i += 1
    return "open" if in_quotes else "closed"


def _logical_records(lines: list[str], delimiter: str):
    """Yield (lineno, text, status) - joining physical lines while a quoted field is open.

    status is "closed" for a well-formed record, otherwise the reason the
    *first* physical line was rejected; parsing then resumes on the next line.
    """
    i = 0
    while i < len(lines):
        buf, j = lines[i], i
        state = _quote_state(buf, delimiter)
        while state == "open" and j + 1 < len(lines) and j - i < MAX_CONTINUATION_LINES:
            j += 1
            buf += "\n" + lines[j]
            state = _quote_state(buf, delimiter)
        if state == "closed":
            yield i + 1, buf, "closed"
            i = j + 1
        else:
            reason = "malformed_quote" if j == i else "unterminated_quote"
            yield i + 1, lines[i], reason       # reject this physical line only
            i += 1


def _split_fields(record: str, delimiter: str) -> list[str]:
    reader = csv.reader([record], delimiter=delimiter, quotechar='"', doublequote=True, strict=False)
    return next(reader, [])


def parse_bytes(raw: bytes, delimiter: str = "\t") -> tuple[list[dict], ParseReport]:
    text, enc, fallback = decode_bytes(raw)
    report = ParseReport(encoding=enc, encoding_fallback=fallback)
    if fallback:
        logger.warning("input is not valid UTF-8; decoded as %s", enc)

    lines = _LINE_BREAK.split(text)
    if lines and lines[-1] == "":
        lines.pop()                             # trailing newline at EOF

    header: list[str] | None = None
    records: list[dict] = []
    for lineno, rec, status in _logical_records(lines, delimiter):
        if header is not None and status != "closed" and rec.strip() and not rec.lstrip().startswith("#"):
            report.total += 1
            report.rejected.append((lineno, status, rec[:200]))
            continue
        stripped = rec.translate(_ZERO_WIDTH).strip()
        if not stripped:
            report.blank += 1
            continue
        if stripped.startswith("#"):
            report.comments += 1
            continue
        if header is None:
            header = [clean_cell(c).lower() for c in _split_fields(rec, delimiter)]
            missing = [c for c in REQUIRED if c not in header]
            if missing:
                raise ValueError(f"header missing required columns {missing}: {header}")
            if len(set(header)) != len(header):
                raise ValueError(f"duplicate column names in header: {header}")
            continue

        report.total += 1
        cells = [clean_cell(c) for c in _split_fields(rec, delimiter)]
        if len(cells) != len(header):
            report.rejected.append((lineno, f"arity: {len(cells)} != {len(header)}", rec[:200]))
            continue
        if any(len(c) > MAX_FIELD_CHARS for c in cells):
            report.rejected.append((lineno, "field_too_long", rec[:200]))
            continue
        row = dict(zip(header, cells))
        empty = [k for k in REQUIRED if not row.get(k)]
        if empty:
            report.rejected.append((lineno, f"missing_required: {empty}", rec[:200]))
            continue
        for k in OPTIONAL:
            row.setdefault(k, "")
        records.append(row)
        report.ok += 1

    assert report.ok + len(report.rejected) == report.total, "parser lost rows"
    if report.rejected:
        logger.warning("rejected %d/%d rows", len(report.rejected), report.total)
    return records, report


def parse_file(path: str | Path, delimiter: str = "\t") -> tuple[list[dict], ParseReport]:
    return parse_bytes(Path(path).read_bytes(), delimiter=delimiter)

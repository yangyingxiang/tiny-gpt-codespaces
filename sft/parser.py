"""Parser for the raw SFT exports in data/raw/.

    records, report = parse_file("data/raw/sft_export_2024.tsv")

Each record is a dict keyed by the header's column names
(id, instruction, input, output, source).

NOTE: this is the quick first version, written against a small, clean sample
export. The real exports are messier. Part 1 of EXERCISE.md is to replace it
with a robust implementation that meets the spec there (tests/test_parser.py).
Keep the public API: `parse_bytes`, `parse_file`, `ParseReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


def parse_bytes(raw: bytes, delimiter: str = "\t") -> tuple[list[dict], ParseReport]:
    text = raw.decode("utf-8-sig", errors="replace")
    report = ParseReport()
    header = None
    records = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            report.blank += 1
            continue
        if line.startswith("#"):
            report.comments += 1
            continue
        cells = line.split(delimiter)
        if header is None:
            header = cells
            continue
        if len(cells) != len(header):
            continue
        records.append(dict(zip(header, cells)))
        report.ok += 1
    report.total = report.ok
    return records, report


def parse_file(path: str | Path, delimiter: str = "\t") -> tuple[list[dict], ParseReport]:
    return parse_bytes(Path(path).read_bytes(), delimiter=delimiter)

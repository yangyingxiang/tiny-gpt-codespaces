"""Part 1 -- the parser contract.

On `main` most of these FAIL: the parser there is a quick first draft.
Your job in Part 1 is to make them pass (and to add the tests you think are
missing). See EXERCISE.md for the full format spec.
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from sft.config import RAW_FILES
from sft.parser import parse_bytes, parse_file

H = "id\tinstruction\tinput\toutput\tsource\n"


def parse(text: str, encoding: str = "utf-8"):
    return parse_bytes(text.encode(encoding))


def test_simple_rows():
    rows, rep = parse(H + "1\tReverse:\tabc\tcba\tvendor_a\n2\tUppercase:\tab\tAB\tvendor_b\n")
    assert [r["id"] for r in rows] == ["1", "2"]
    assert rows[0] == {"id": "1", "instruction": "Reverse:", "input": "abc", "output": "cba", "source": "vendor_a"}
    assert (rep.total, rep.ok, rep.rejected) == (2, 2, [])


def test_utf8_bom_is_not_part_of_first_column_name():
    rows, _ = parse_bytes(("﻿" + H + "1\tReverse:\tabc\tcba\tx\n").encode("utf-8"))
    assert "id" in rows[0] and rows[0]["id"] == "1"


def test_column_order_comes_from_header():
    rows, _ = parse("output\tid\tinstruction\n42\t7\tLength:\n")
    assert rows[0]["id"] == "7" and rows[0]["output"] == "42" and rows[0]["instruction"] == "Length:"


def test_missing_optional_columns_default_to_empty_string():
    rows, _ = parse("id\tinstruction\toutput\n1\tSay hi:\thi\n")
    assert rows[0]["input"] == "" and rows[0]["source"] == ""


def test_header_without_required_column_raises():
    with pytest.raises(ValueError):
        parse("id\tinstruction\tinput\n1\ta\tb\n")


def test_mixed_line_endings():
    rows, rep = parse(H + "1\ta\tb\tc\td\r\n2\ta\tb\tc\td\r3\ta\tb\tc\td\n")
    assert [r["id"] for r in rows] == ["1", "2", "3"]
    assert all(r["source"] == "d" for r in rows)          # no stray '\r'


def test_blank_and_comment_lines_are_skipped_and_counted():
    rows, rep = parse("# export v3\n\n" + H + "\n1\ta\tb\tc\td\n# page break\n   \n2\ta\tb\tc\td\n")
    assert len(rows) == 2
    assert rep.total == 2 and rep.comments == 2 and rep.blank == 3


def test_quoted_fields_with_tab_newline_and_doubled_quotes():
    text = H + '1\tRepeat:\t"she said ""hi"""\t"a\tb\nc"\tx\n2\ta\tb\tc\td\n'
    rows, rep = parse(text)
    assert rows[0]["input"] == 'she said "hi"'
    assert rows[0]["output"] == "a\tb\nc"
    assert rows[1]["id"] == "2"
    assert rep.ok == 2


def test_quote_in_the_middle_of_a_field_is_literal():
    rows, _ = parse(H + '1\tRepeat:\ta 5" screen\ta 5" screen\tx\n')
    assert rows[0]["input"] == 'a 5" screen'


def test_ragged_rows_are_rejected_with_a_reason_not_dropped():
    rows, rep = parse(H + "1\ta\tb\tc\td\n2\ta\tb\n3\ta\tb\tc\td\tEXTRA\n4\ta\tb\tc\td\n")
    assert [r["id"] for r in rows] == ["1", "4"]
    assert rep.total == 4 and rep.ok == 2 and len(rep.rejected) == 2
    assert {lineno for lineno, _, _ in rep.rejected} == {3, 4}
    assert all("arity" in reason for _, reason, _ in rep.rejected)


def test_empty_required_field_is_rejected():
    rows, rep = parse(H + "1\tReverse:\tabc\t\tx\n2\t\tabc\tcba\tx\n3\tReverse:\t\tcba\tx\n")
    assert [r["id"] for r in rows] == ["3"]                 # empty *input* is allowed
    assert len(rep.rejected) == 2
    assert all("missing" in reason for _, reason, _ in rep.rejected)


def test_cells_are_stripped_nfc_normalised_and_zero_width_free():
    nfd = unicodedata.normalize("NFD", "café")
    rows, _ = parse(H + f"1\t  Repeat:  \t{nfd}​\t{nfd}\tx\n")
    assert rows[0]["instruction"] == "Repeat:"
    assert rows[0]["input"] == "café" and len(rows[0]["input"]) == 4


def test_cp1252_input_falls_back_and_is_flagged():
    text = "id\tinstruction\tinput\toutput\n1\tRepeat:\tdéjà ’n’ vu\tdéjà ’n’ vu\n"
    rows, rep = parse(text, encoding="cp1252")
    assert rows[0]["input"] == "déjà ’n’ vu"
    assert rep.encoding_fallback is True


def test_unterminated_quote_does_not_swallow_the_rest_of_the_file():
    text = H + '1\ta\t"never closed\tc\td\n' + "".join(f"{i}\ta\tb\tc\td\n" for i in range(2, 12))
    rows, rep = parse(text)
    assert [r["id"] for r in rows] == [str(i) for i in range(2, 12)]
    assert len(rep.rejected) == 1 and rep.rejected[0][0] == 2


def test_empty_and_header_only_inputs():
    assert parse_bytes(b"")[0] == []
    rows, rep = parse(H)
    assert rows == [] and rep.total == 0


def test_accounting_invariant_on_the_real_exports():
    for path in RAW_FILES:
        rows, rep = parse_file(path)
        assert rep.ok == len(rows)
        assert rep.ok + len(rep.rejected) == rep.total
        assert rep.ok > 0.9 * rep.total, rep.summary()
        for r in rows:
            assert r["id"] and r["instruction"] and r["output"]
            assert "�" not in r["input"] + r["output"], r     # no mojibake
            assert not r["input"].startswith('"'), r               # quotes were unwrapped


# --------------------------------------------------------------------------- property tests

ALPHABET = list("abcXYZ019 \t\n\"'#,;") + ["é", "ü", "’", "日", " "]


def _rand_cell(rng: random.Random) -> str:
    s = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
    s = unicodedata.normalize("NFC", s).strip()
    return s or "x"


def _to_tsv_field(s: str) -> str:
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in '\t\n\r"') else s


def test_roundtrip_random_records():
    rng = random.Random(0)
    for _ in range(200):
        recs = [
            {"id": str(i), "instruction": _rand_cell(rng), "input": _rand_cell(rng),
             "output": _rand_cell(rng), "source": _rand_cell(rng)}
            for i in range(rng.randint(1, 5))
        ]
        # a leading '#' would make the line a comment; keep ids first so that can't happen
        body = "".join("\t".join(_to_tsv_field(r[k]) for k in ("id", "instruction", "input", "output", "source")) + "\n"
                       for r in recs)
        rows, rep = parse(H + body)
        assert rows == recs, (H + body)


def test_random_bytes_never_crash():
    rng = random.Random(1)
    for _ in range(300):
        junk = bytes(rng.randrange(256) for _ in range(rng.randint(0, 200)))
        raw = H.encode() + junk
        rows, rep = parse_bytes(raw)                       # must not raise
        assert rep.ok + len(rep.rejected) == rep.total

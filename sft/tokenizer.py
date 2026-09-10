"""Byte-level tokenizer with a handful of chat special tokens.

Ids 0..255 are raw UTF-8 bytes, so any Unicode string round-trips. Special
tokens are appended after the byte range.
"""

from __future__ import annotations

NUM_BYTES = 256
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]
SPECIAL_IDS = {tok: NUM_BYTES + i for i, tok in enumerate(SPECIAL_TOKENS)}

PAD_ID = SPECIAL_IDS["<|pad|>"]
BOS_ID = SPECIAL_IDS["<|bos|>"]
EOS_ID = SPECIAL_IDS["<|eos|>"]
USER_ID = SPECIAL_IDS["<|user|>"]
ASSISTANT_ID = SPECIAL_IDS["<|assistant|>"]


class ByteTokenizer:
    vocab_size = NUM_BYTES + len(SPECIAL_TOKENS)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special: bool = True) -> str:
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i < NUM_BYTES:
                buf.append(i)
                continue
            out.append(buf.decode("utf-8", errors="replace"))
            buf = bytearray()
            if not skip_special:
                out.append(SPECIAL_TOKENS[i - NUM_BYTES])
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def id_to_token(self, i: int) -> str:
        if i < NUM_BYTES:
            return repr(bytes([i]))
        return SPECIAL_TOKENS[i - NUM_BYTES]

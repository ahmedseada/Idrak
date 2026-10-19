# Copyright (c) 2026 Ahmed Seada
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Plan 24, T3: Hugging Face tokenizers' detailed encodings, as references for Idrak's TokenizerEncoding.

Writes tests/Idrak.Tests/data/tokenizers/encodings.json from the pinned tokenizers (tools/tokenizers/requirements.txt):

- "byte_level": for BPE tokenizer.json files in the test data (byte-level and SentencePiece-style; two derived here from
  pretrain/tokenizer with the ByteLevel and RobertaProcessing post-processors, which trim offsets), every line of
  tokenizers/corpus.txt encoded with special tokens: ids, tokens, offsets (UTF-16), type ids, special-tokens mask,
  sequence ids (-1 for special tokens).
- "truncation": wordpiece-uncased pairs truncated to 16 tokens by each strategy (a strategy tokenizers refuses for a pair
  is left out).
- "padding": a batch of four lines padded to the longest, and padded to 24 tokens.

Nothing is downloaded. Reruns write the same bytes. Run with the reference venv:
    python -I tools/tokenizers/make_encodings.py
"""

import json
import os
import sys

import tokenizers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "tests", "Idrak.Tests", "data")
BYTE_LEVEL = [
    "pretrain/tokenizer/tokenizer.json",            # byte level, no post-processor
    "pretrain/metaspace/tokenizer.json",            # SentencePiece-style BPE (Metaspace)
    "vlm-qwen/tiny-qwen2-vl/tokenizer.json",        # Qwen's Split + ByteLevel
    "vlm/tiny-gemma3/tokenizer.json",               # byte fallback, TemplateProcessing
    "audio/tiny-whisper/tokenizer.json",            # Whisper's byte level
    "tokenizers/byte-level-trim/tokenizer.json",    # derived below: ByteLevel post-processor trimming offsets
    "tokenizers/byte-level-roberta/tokenizer.json", # derived below: RobertaProcessing
]


def derive():
    """Two tokenizers made from pretrain/tokenizer with the post-processors that trim offsets, saved by tokenizers."""
    from tokenizers import processors
    source = os.path.join(DATA, "pretrain", "tokenizer", "tokenizer.json")
    trim = tokenizers.Tokenizer.from_file(source)
    trim.post_processor = processors.ByteLevel(trim_offsets=True)
    roberta = tokenizers.Tokenizer.from_file(source)
    roberta.post_processor = processors.RobertaProcessing(("</s>", roberta.token_to_id("</s>")), ("<s>", roberta.token_to_id("<s>")),
                                                           trim_offsets=True, add_prefix_space=False)
    for name, tokenizer in [("byte-level-trim", trim), ("byte-level-roberta", roberta)]:
        folder = os.path.join(DATA, "tokenizers", name)
        os.makedirs(folder, exist_ok=True)
        tokenizer.save(os.path.join(folder, "tokenizer.json"), pretty=True)


def utf16(text):
    """Code-point index -> UTF-16 index (a character outside the BMP counts two)."""
    index = [0]
    for c in text:
        index.append(index[-1] + (2 if ord(c) > 0xFFFF else 1))
    return index


def entry(encoding, text, pair=None):
    a, b = utf16(text), utf16(pair) if pair is not None else None
    offsets = []
    for (start, end), sequence in zip(encoding.offsets, encoding.sequence_ids):
        index = b if sequence == 1 else a
        offsets.append([index[start], index[end]])
    return {
        "ids": encoding.ids,
        "tokens": encoding.tokens,
        "offsets": offsets,
        "type_ids": encoding.type_ids,
        "special_tokens_mask": encoding.special_tokens_mask,
        "attention_mask": encoding.attention_mask,
        "sequence_ids": [-1 if s is None else s for s in encoding.sequence_ids],
    }


def main():
    with open(os.path.join(DATA, "tokenizers", "corpus.txt"), encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")

    derive()
    out = {"tokenizers": tokenizers.__version__, "byte_level": {}, "truncation": [], "padding": {}}
    for path in BYTE_LEVEL:
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(DATA, path))
        out["byte_level"][path] = [dict(entry(tokenizer.encode(line), line), text=line) for line in lines]

    wordpiece = tokenizers.Tokenizer.from_file(os.path.join(DATA, "tokenizers", "wordpiece-uncased", "tokenizer.json"))
    texts = [line for line in lines[:40] if line.strip()]
    for strategy in ["longest_first", "only_first", "only_second"]:
        for i in range(0, 12, 2):
            text, pair = texts[i], texts[i + 1]
            wordpiece.enable_truncation(16, strategy=strategy)
            try:
                encoding = wordpiece.encode(text, pair)
            except Exception:  # tokenizers refuses a strategy that cannot fit: left out
                continue
            out["truncation"].append(dict(entry(encoding, text, pair), text=text, pair=pair, max_length=16, strategy=strategy))
    wordpiece.no_truncation()

    batch = texts[:4]
    wordpiece.enable_padding(pad_id=0, pad_token="[PAD]")
    out["padding"]["longest"] = [dict(entry(e, t), text=t) for e, t in zip(wordpiece.encode_batch(batch), batch)]
    wordpiece.enable_padding(pad_id=0, pad_token="[PAD]", length=24)
    wordpiece.enable_truncation(24)
    out["padding"]["max_length"] = [dict(entry(e, t), text=t) for e, t in zip(wordpiece.encode_batch(batch), batch)]

    target = os.path.join(DATA, "tokenizers", "encodings.json")
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {target}", file=sys.stderr)


if __name__ == "__main__":
    main()

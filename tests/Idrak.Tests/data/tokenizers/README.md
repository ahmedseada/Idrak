# The references for Unigram, SentencePiece and WordPiece tokenizers (plan 24, phase T0)

Written by `tools/tokenizers/make_fixtures.py` from `tools/tokenizers/requirements.txt` (tokenizers 0.23.2,
sentencepiece 0.2.2, transformers 5.18.0, protobuf 7.36.2; Python's Unicode data 15.1.0); `manifest.json` records the
versions and every file's size and SHA-256. Nothing is downloaded: the corpus is written by the script, and the
tokenizers are trained on it plus `../pretrain/corpus.txt` (whose SHA-256 the manifest records too). Two runs write the
same bytes with the same library versions (checked: every file identical).

Offsets everywhere are **UTF-16 indexes** into the text, as C# strings index it: the script converts sentencepiece's byte
offsets and tokenizers' code-point offsets (a character outside the BMP counts two).

| File | What |
|---|---|
| `corpus.txt` | 102 lines written to break tokenizers: Arabic with harakat, shadda and dagger alef, tatweel (alone and inside words), mixed directions with Western, Arabic-Indic and Persian digits and bidi marks (LRM, RLM, embeddings, isolates), emoji with skin tones, flags (regional indicators and a tag sequence), ZWJ sequences and keycaps, CJK, fullwidth and halfwidth forms, characters outside the BMP, accents precomposed and decomposed, stacked combining marks, Devanagari, Thai, Hebrew with niqqud, ligatures and compatibility characters, runs of spaces, tabs, NBSP and other spaces, ZWSP/ZWJ/ZWNJ, a BOM, CRLF lines, empty and blank lines, very long words (300 `a`s, a 120-digit number), C0 and C1 control characters, U+2028/U+2029, special-token text, then ordinary prose in 20 languages. **A line is the file split at LF** (the final LF ends the last line), so a CRLF line keeps its CR |
| `unigram/` | SentencePiece Unigram, 1,000 pieces, T5's layout (pad 0, eos 1, unk 2, no bos), control symbols `<cls>` `<sep>`, user-defined `<mask>`, the `nmt_nfkc` charsmap (179,200 trie bytes) |
| `unigram-bytes/` | SentencePiece Unigram with byte fallback (256 `<0xNN>` pieces), digits split, and a user rule compiled to a small charsmap (`user_defined`, 30 entries: tatweel removed, Arabic-Indic and Persian digits to ASCII, alef and yeh forms, two ligatures, `AA` → `aa`) |
| `bpe/` | SentencePiece BPE, Llama-2's layout: byte fallback, digits split, `identity` normalization (no charsmap), extra whitespace kept, whitespace-only pieces allowed |
| `wordpiece-uncased/`, `wordpiece-cased/` | WordPiece, 1,861 and 2,472 pieces (see below), as `BertTokenizerFast` writes them |
| `known-differences.json` | the corpus lines where sentencepiece and tokenizers disagree on the same model (below) |
| `real/` | written only by `--real` (the owner's run, below): the real models' references |

Each SentencePiece folder holds:

| File | What |
|---|---|
| `tokenizer.model` | the model as `sentencepiece` trained it (one thread, no shuffling, seed 2026, from an iterator, so no file path is stored; the user rule's file is stored by its bare name, `arabic-rules.tsv`) |
| `tokenizer.json` | what transformers' converters make of it, the way the Hub's files were made: `unigram` through `T5Converter` (2 `<extra_id_N>` sentinels, `</s>` appended), `bpe` through `LlamaConverter` (legacy, prefix space; `<s>` prepended as `LlamaTokenizerFast` does), `unigram-bytes` through `SpmConverter` with byte fallback kept and Llama's decoder (no Hub class keeps a Unigram model's byte fallback: T5's and ALBERT's converters drop it); then the special tokens the fast class adds |
| `tokenizer_config.json` | the slow class and special tokens a Hub folder would carry |
| `model.json` | the `ModelProto` as the official protobuf (`sentencepiece_model_pb2`) reads it: every piece with its score (and the float's bits) and type; every `TrainerSpec` and `NormalizerSpec` field with the proto's default when absent (floats with `_bits`, bytes as size and SHA-256, `present` naming the fields in the file); `charsmap`: the precompiled charsmap's trie decoded by an independent Python reader of darts-clone units (units, entries, the SHA-256 of the sorted listing `hex(key)\thex(normalized)\n`, the longest key) |
| `expected-spm.json` | `sentencepiece` on every corpus line: ids, pieces, offsets, type ids (0), special-tokens mask (control pieces), decoded text; no BOS or EOS (sentencepiece adds none by default) |
| `expected-hf.json` | `tokenizers` on `tokenizer.json` for every line (with the post-processor's special tokens): ids, pieces, offsets, type ids, special-tokens mask, sequence ids (-1 for special tokens), decoded text with and without special tokens; and `pairs`: consecutive non-empty lines from the first 24, encoded as pairs (the pair's offsets index its own text) |
| `normalized.json` | per line: `precompiled`, the charsmap alone through tokenizers' `Precompiled`; `spm`, sentencepiece's whole normalizer (charsmap, dummy prefix, extra whitespace, `▁`) and `spm_offsets`, for each code point of its output the UTF-16 index in the line where it starts, then the end |

The WordPiece folders hold `vocab.txt` (one token a line, by id), `tokenizer_config.json` (`BertTokenizer`'s),
`tokenizer.json` made by transformers' `BertConverter` (`BertNormalizer` with Chinese characters split and accents
following lowercase, `BertPreTokenizer`, `WordPiece` with `[UNK]` and 100 characters a word, the `WordPiece` decoder;
`wordpiece-uncased` keeps `BertConverter`'s `TemplateProcessing`, `wordpiece-cased` has tokenizers' `BertProcessing`, as
older BERT files do), and `expected-hf.json`.

**Why the WordPiece vocabularies are not from tokenizers' `WordPieceTrainer`:** it is not reproducible. It numbers the
`##` alphabet in Rust's hash order, which changes every process, and merges between equal counts are broken by those
numbers: two runs of 0.23.2 on this corpus learned 39 different pieces. So the script runs the words `BertNormalizer` and
`BertPreTokenizer` make of the training text through a SentencePiece BPE run (deterministic): a word-initial piece `▁x`
becomes `x`, any other becomes `##x`, every alphabet character is kept in both forms, and `[PAD] [UNK] [CLS] [SEP] [MASK]`
come first. The tokenizer and its references are still tokenizers' own.

## Where sentencepiece and tokenizers disagree (`known-differences.json`)

21 lines; each path is held to its own reference. The kinds:

- **Unknown pieces decode differently** (`unigram`, same ids): sentencepiece writes ` ⁇ ` for `<unk>`, tokenizers skips it
  as a special token.
- **Viterbi ties** (`unigram`, 300 `a`s): the two libraries split a run of equal-score pieces at different places.
- **Special-token text** (all three): tokenizers matches `</s>`, `<unk>`, `<cls>` written in the text as added tokens;
  sentencepiece never matches control symbols in text.
- **Precompiled by grapheme** (`unigram-bytes`): tokenizers' `Precompiled` replaces a whole grapheme cluster by the first
  key that is a prefix of it, so the mark after a mapped letter is lost: `أَ` (alef with hamza, fatha) becomes `ا`, and
  `ی` + ZWNJ becomes `ي`; sentencepiece maps the letter and keeps the mark.
- **Trailing tabs and CR** (`unigram-bytes`): `SpmConverter`'s `Strip` removes them, sentencepiece keeps them as byte
  pieces (its `nmt_nfkc` maps them to spaces, so `unigram` agrees).

## How the C# tests read it

- `"tokenizers: sentencepiece"`: each `tokenizer.model` reads to `model.json` exactly (pieces, scores bit for bit, types,
  every trainer and normalizer field, the charsmap's size and SHA-256); truncated and corrupt copies fail naming the byte.
- `"tokenizers: trie"`: each charsmap's trie reads to `model.json`'s `charsmap` (entry count and listing SHA-256), and its
  common-prefix search at every byte of every corpus line equals a scan of the decoded entries.
- Later phases: Precompiled against `normalized.json`, Unigram and SentencePiece BPE against `expected-spm.json` (from
  `tokenizer.model`) and `expected-hf.json` (from `tokenizer.json`), WordPiece against `expected-hf.json` (from
  `tokenizer.json` and from `vocab.txt`).

## Regenerate

```
python -m venv tok && tok/bin/pip install -r tools/tokenizers/requirements.txt
tok/bin/python -I tools/tokenizers/make_fixtures.py
```

From `D:\Projects\Idrak` on Windows:

```powershell
python -m venv $env:USERPROFILE\tok
& $env:USERPROFILE\tok\Scripts\python.exe -m pip install -r tools\tokenizers\requirements.txt
& $env:USERPROFILE\tok\Scripts\python.exe -I tools\tokenizers\make_fixtures.py
```

Everything in this folder but this README and `real/` is removed first.

## The real models (`--real`, the owner's)

```powershell
& $env:USERPROFILE\tok\Scripts\python.exe -I tools\tokenizers\make_fixtures.py --real
```

Downloads the tokenizer files of google-t5/t5-small, google/mt5-small, albert/albert-base-v2,
FacebookAI/xlm-roberta-base, hf-internal-testing/llama-tokenizer, google-bert/bert-base-uncased,
google-bert/bert-base-multilingual-cased and aubmindlab/bert-base-arabertv02 into `<cache>\tokenizers\<org>--<name>\`
(the cache is `IDRAK_CACHE`, else `~/.cache/idrak`; `--cache` overrides it) and writes only references:
`real/<name>.json` with the files' sizes and SHA-256, the versions, tokenizers' references on every corpus line (from
`tokenizer.json`, or transformers' `AutoTokenizer` when a folder has only `vocab.txt`), and for SentencePiece models the
`ModelProto` summary, sentencepiece's references and the differences. The model files themselves are not committed.

## Speed (`--bench`)

```powershell
& $env:USERPROFILE\tok\Scripts\python.exe -I tools\tokenizers\make_fixtures.py --bench
```

Prints, as JSON, each reference's tokens a second on the corpus repeated to 200,000 characters: sentencepiece and
tokenizers, one thread (one encode call a text) and all threads (`num_threads`, `encode_batch`), the first pass (cold)
apart from the best later one (warm). `--seconds` sets how long each measurement runs.

# Plan 24: Unigram (SentencePiece) and WordPiece tokenizers, built in

**Status (2026-10-19, `abstraction` 7b8c85f):** planned 2026-10-19 from the owner's spec (plan 24, draft 1) against
e53acb8. Every phase but T5 is built, merged and pushed, and so are the handover's three leftovers (library-owned
Unicode normalization, the T3 callers, the T4 leftovers) and two speed rounds the `--bench-text` rows exposed; the last
merged run passed 147 of 147 targeted CPU tests. T5 was measured and not taken (same ids, about half the speed); the owner's Windows and CUDA runs are
next ("The owner's runs"). Closes plug-in gap 11 ([plug-in.md](plug-in.md), "Only BPE tokenizer models").

| Phase | What | State |
|---|---|---|
| L0 | The tokenizer model registry, one tokenizer contract, one loader; callers off `BpeTokenizer` | Done; CPU pass (155 tests) |
| L1 | The span pipeline with alignments: built-in components on spans, the post-processor table | Done; CPU pass |
| T0 | Reference fixtures; the SentencePiece `tokenizer.model` reader; the Precompiled normalizer | Done; CPU pass |
| T1 | Unigram: encode, decode, byte fallback, sampling | Done; CPU pass |
| T2 | WordPiece: the BERT components, greedy longest match, `[CLS] … [SEP]` and pairs | Done; CPU pass |
| T3 | Encodings (offsets, type ids, masks, pairs, batch), detection across files, GGUF `llama` / `bert` / `t5` | Done; CPU pass (BPE, WordPiece, Unigram); callers done after the handover ("Finished after the handover", 2) |
| T4 | Trainers: Unigram (EM), WordPiece; `idrak data tokenizer --model` | Done; CPU pass (unigram: SentencePiece's pieces, 100% on the fixtures; wordpiece: exact against the fixed-tie reference); leftovers done after the handover (3) |
| N | Unicode normalization owned by the library (`UnicodeNormalization`, Unicode 18.0 tables) | Done after the handover (1); CPU pass, a real CLI process checked |
| S | Speed rounds from the `--bench-text` rows: BPE, the encodings, the trainers, regex contention, WordPiece | Done (4, 5); results proven unchanged by full encoding dumps |
| T5 | BPE onto the shared span pipeline (only if measured equal or faster) | Measured 2026-10-19: same ids, about half the speed; not taken |

## Goal

Load and run, out of the box, every tokenizer the common model families ship, not just BPE: Unigram (the model inside
SentencePiece: T5, mT5, ALBERT, XLNet, XLM-R, many multilingual models) and WordPiece (BERT, most embedding and
classification models). BPE, Unigram and WordPiece are **library defaults in one registry**, registered through the same
`Register` call an app uses, so they work with no setup and stay overridable like every registered implementation (the
owner, 2026-10-19: built into the library, not a separate package). Then train both, as the BPE trainer does.

Scope: tokenizers only. The library has no BERT or T5 architecture; those model families are plug-ins (CLAUDE.md:
families live in plug-ins or apps), not part of this plan. What this plan makes run end to end is every model whose
architecture already exists and whose tokenizer was refused (a Llama-2-style GGUF with `tokenizer.ggml.model = llama`),
and `idrak tokenize` on any T5, ALBERT, XLM-R or BERT folder.

## Decisions (taken by the owner, 2026-10-19)

1. **Inside the library**, not a separate package: the three models and their components are registered defaults in
   Idrak, so `PretrainedModel.Load` and GGUF loading read T5 and BERT tokenizers with nothing added.
2. **The public API may break in L0**: `ModelFiles.Tokenizer`, `ITokenizerTrainer.Train`, `BaseModelInfo` and the
   `is BpeTokenizer` checks move to the new contract. `BpeTokenizer` keeps every public member. No version bump, no
   CHANGELOG line (CLAUDE.md), whatever the API test's message says.
3. **Fixtures**: small tokenizers trained with the reference libraries, plus their reference outputs, are committed. The
   real models' tokenizer files are not committed (4-16 MB each): the fixture script downloads them into the cache and
   writes only their reference outputs, and their tests are a separate group the owner runs (`tokenizers real:`).

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../docs/optimization.md) apply in full. The ones that shape this plan:

- **Contract plus registry (decision 10).** Every model, file reader, component and post-processor is a registered
  slot; nothing dispatches on a type name in a `switch`. A contract any package can use lives in `Idrak.Abstraction`;
  one only Idrak's tokenizer loading uses lives in `Idrak.Models.Abstractions`, beside `TokenizerComponents`.
- **No native libraries.** No `sentencepiece` or `tokenizers` binaries, no protobuf package: the formats are read in C#
  (rule 0, rule 70). The Python libraries are used only by the fixture script, as references.
- **Bytes and spans on the hot path** (rules 1, 11, 74): files are parsed from bytes, encoding runs on spans over
  pooled buffers returned in `finally`, vocabularies are looked up through alternate lookups or a trie, never by
  building a string per piece. After warm-up, encoding into a caller's list allocates nothing (measured, rule 63).
- **The same result, proven** (rule 71): every id and offset equals the reference's; BPE's ids do not change by one on
  any existing fixture; batch encoding equals single encoding.
- **The reference is not the ceiling** (CLAUDE.md): `tokenizers` and `sentencepiece` fix the ids, offsets and texts,
  never the speed or memory. Where `--bench-text` or a profile shows a candidate that beats them (a faster trie walk,
  fewer passes, less memory per vocabulary), it is taken, with the ids proven unchanged.
- **Nothing fixed** (CLAUDE.md): every size, limit and choice is a `TokenizerOptions` setting with an `IDRAK_TOKENIZER_*`
  variable, carried through `PretrainedOptions`, the CLI and the benches (see "Settings").
- **Use every core** unless the user sets a limit: batch encoding and training spread work over `Workers` (0: every
  core), in a fixed combination order so the result does not depend on the worker count (rule 79, rule 86's idea).
- **Tests only through filters**, CPU here: `IDRAK_DEVICES=cpu IDRAK_FILTER="tokenizers:"` (and the group named in each
  phase). The owner's runs are listed per phase, as exact PowerShell commands.
- **Dated runs** go in the Runs table at the end; each phase gets an "as built" note when merged.

## What exists (2026-10-19, e53acb8)

- `ITokenizer` (`src/Idrak.Abstraction/Generation/Tokenizer.cs:10`): `VocabularySize`, `Encode`, `Decode`, `TokenOf`.
  Too thin for model loading, so callers use `BpeTokenizer` directly for `PadVocabulary`, `IdOf`, `IsSpecial`,
  `ToJson` / `Save` / `WriteTo` and `EncodeRange`.
- `BpeTokenizer` (`src/Idrak/Models/BpeTokenizer.cs`, 1,890 lines) holds the BPE model and the whole pipeline with its
  fast paths; it refuses any model type but `"BPE"` (`:94`, no Register hint). About 40 files name it
  (`PretrainedModel.cs:47` `required BpeTokenizer`, `:381` `BpeTokenizer.Load`; FineTuning, SpeechModels, ServeHost,
  ModelFacts, BaseModelInfo, the Create, Text, Data and Check commands, samples, tests).
- `TokenizerComponents` (`src/Idrak/Models/Abstractions/TokenizerComponents.cs`): three `SlotTable` registries, built-in
  names: normalizers Sequence, NFC, NFKC, NFD, NFKD, Prepend, Replace, Lowercase; pre-tokenizers Sequence, Split,
  ByteLevel, Metaspace, Digits, Whitespace; decoders Sequence, ByteLevel, Replace, ByteFallback, Fuse, Strip,
  Metaspace. The contracts are string in, string out (`Normalize(string)`, `PreTokenize(IReadOnlyList<string>, bool)`):
  no alignments, so no offsets. There is no model table and no post-processor table.
- Encoding returns ids only: no offsets, type ids, masks, pairs, truncation, padding or batch.
- GGUF (`GgufModel.WriteTokenizer`, `GgufModel.cs:268`) writes a tokenizer.json for `gpt2` only and refuses every other
  `tokenizer.ggml.model`; integer metadata arrays come back as `long[]` (`Gguf.cs:651`), so a uint8 charsmap would take
  8 bytes a byte.
- `TokenizerTrainers` has `"bpe"` (`BpeTrainer`, exact against Hugging Face's `BpeTrainer`); `ITokenizerTrainer.Train`
  returns `BpeTokenizer`; `idrak data tokenizer` picks a trainer with `--trainer`.
- Downloads already fetch `tokenizer.model` and `vocab.txt` (`HuggingFaceModels.cs:22`); nothing reads them.
- The only protobuf code is the ONNX reader (`src/Idrak/Onnx/OnnxModel.cs`).
- Tests: `tokenizer training:`, `pretrained:` (BPE fixtures), `plugins:` (registered components;
  `ModelPluginTests.cs:71` asserts 7 decoder types), `gguf:`, `TokenizerSuite` in `Idrak.Abstraction.Testing`.

## Where it lives

| What | Where |
|---|---|
| `IPretrainedTokenizer`, `TokenizerEncoding`, `TokenizerEncodeOptions` | `src/Idrak.Abstraction/Generation/Tokenizer.cs` (namespace `Idrak.Abstraction.Generation`, beside `ITokenizer`): Idrak, Idrak.Nlp, Idrak.Audio, the CLI and samples hold tokenizers |
| `TokenizerModels`, `TokenizerFiles`, `TokenizerDefinition`, `TokenizerOptions`, `NormalizedText`, `PreTokenizedText`, post-processor contract and table | `src/Idrak/Models/Abstractions/` (namespace `Idrak.Models.Abstractions`, beside `TokenizerComponents` and `TokenizerTrainers`) |
| The built-in models, readers, components, trainers | `src/Idrak/Models/Tokenizers/` (namespace `Idrak.Models`); `BpeTokenizer.cs` and `BpeTrainer.cs` stay where they are |
| Library defaults | `src/Idrak/Models/LibraryModelFormats.cs` (`RegisterTokenizerModels`, `RegisterTokenizerFiles`), called from each registry's static constructor through `Overrides.AsLibraryDefaults`, as `GgufPreTokenizers` does |
| Contract suites | `src/Idrak.Abstraction.Testing/Contracts/TokenizerSuite.cs` (extended), `TokenizerModelSuite.cs` (new) |
| Tests | `tests/Idrak.Tests/TokenizerTests.cs` (`tokenizers:`), `TokenizerRealTests.cs` (`tokenizers real:`), `TokenizerTrainingTests.cs` (extended) |
| Fixtures | `tests/Idrak.Tests/data/tokenizers/` from `tools/tokenizers/make_fixtures.py` |

New files under `src/Idrak/Models/Tokenizers/`: `PretrainedTokenizers.cs` (the loader), `TokenizerPipeline.cs` (the
span pipeline), `SpanComponents.cs` (built-in components on spans), `DoubleArrayTrie.cs`, `ProtoReader.cs`,
`SentencePieceModel.cs`, `PrecompiledNormalizer.cs`, `UnigramTokenizer.cs`, `WordPieceTokenizer.cs`,
`BertComponents.cs`, `PostProcessors.cs`, `UnigramTrainer.cs`, `WordPieceTrainer.cs`, `SentencePieceWriter.cs`.

## Conventions every phase follows

The conventions of [plan 19](19-vision-language-from-scratch.md#conventions-every-phase-follows-read-before-coding)
hold: registries as `RandomTextSources` (`SlotTable` named after the class, `RegisterDefault` inside
`Overrides.AsLibraryDefaults`, `[MethodImpl(NoInlining)] Register` with the calling assembly, `Unregister`, `Names`,
`Find`, `Get`, `Default`, `Origin`, `Describe`), `Get`'s message `"{Kind} '{name}' is not registered (registered:
{list}); add it with {Registry}.Register."` (the CLI appends the `-P` hint), a contract suite per registry, test groups
as `(string Name, Action<Device> Run)[]` added to `Tests.All`, registries left as found (`Unregister` in `finally`),
commands with `Usage`, an Arabic summary and `EnvironmentVariables.cs` entries, regeneration of the inventory and the API
dump after a public change (no CHANGELOG line).

Fixtures follow plan 19's rule: reruns write the same bytes (fixed seeds, one thread, sorted JSON, the output folder
removed first), a `manifest.json` with the library versions and each file's size and SHA-256, and a hand-written
README. Ids and offsets are compared exactly.

**Offsets are UTF-16 indexes** into the C# string. Python's are code-point indexes; the fixture script converts them
(a character outside the BMP counts two), so an emoji's offsets compare exactly.

## Settings

`TokenizerOptions` (a record in `Idrak.Models.Abstractions`), given to `PretrainedTokenizers.Load(path, options)` and
through `PretrainedOptions.Tokenizer`; a null field takes its variable, then the default. The CLI lists the variables in
`EnvironmentVariables.cs`; `--bench-text` reads the same ones.

| Setting | Variable | Default | Phase |
|---|---|---|---|
| `Format`: which file to read when a folder has several (`tokenizer.json`, `sentencepiece`, `vocab.txt`, or a registered reader) | `IDRAK_TOKENIZER_FORMAT` | the first that exists, in that order | L0 |
| `PieceCacheLimit`: cached encoded pieces (BPE's fixed 100,000 today, Unigram's and WordPiece's word caches) | `IDRAK_TOKENIZER_CACHE` | 100,000 | L0 |
| `Workers`: threads for batch encoding (0: every core) | `IDRAK_TOKENIZER_WORKERS` | 0 | T3 |
| `SamplingAlpha`, `SamplingBest`, `SamplingSeed`: Unigram subword regularization (off when alpha is null) | `IDRAK_TOKENIZER_ALPHA`, `IDRAK_TOKENIZER_NBEST`, `IDRAK_TOKENIZER_SEED` | off | T1 |
| `MaxInputCharsPerWord`: WordPiece's limit, over the file's | `IDRAK_TOKENIZER_MAX_WORD_CHARS` | the file's (100) | T2 |

The trainers' settings are `TokenizerTrainingOptions` fields with CLI options (T4).

---

## Phase L0: the model registry, one contract, one loader

**Why first.** Nothing else can load while every caller requires `BpeTokenizer` and the constructor refuses any other
model type.

**Contract** (`src/Idrak.Abstraction/Generation/Tokenizer.cs`):

```csharp
public interface IPretrainedTokenizer : ITokenizer
{
    string Model { get; }                                   // "BPE", "Unigram", "WordPiece", or a registered type
    int? IdOf(string token);
    bool IsSpecial(int id);
    void PadVocabulary(int size);
    void EncodeRange(string text, int start, int length, List<int> ids);   // appends; allocates nothing once warm
    JsonObject ToJson();                                    // tokenizer.json
    void WriteTo(Stream stream);
    void Save(string path);
}
```

`BpeTokenizer` implements it with the members it already has; its encode and decode code does not change.

**Registry** (`src/Idrak/Models/Abstractions/TokenizerModels.cs`):

```csharp
public sealed class TokenizerDefinition                    // what a reader gives a model
{
    public required string Source { get; init; }           // the file, for messages
    public required JsonObject Pipeline { get; init; }     // tokenizer.json without model.vocab / model.merges
    public required JsonObject Model { get; init; }        // model.* without the big tables
    public required TokenTable Vocabulary { get; init; }   // pieces, scores (double), types, read straight from bytes
    public (string Left, string Right)[]? Merges { get; init; }
}

public static class TokenizerModels
{
    public static void Register(string type, Func<TokenizerDefinition, TokenizerOptions, IPretrainedTokenizer> create);
    public static IPretrainedTokenizer Create(TokenizerDefinition definition, TokenizerOptions? options = null);
    // Unregister, Names, Find, Get, Default, Origin, SetPolicy, Describe as the convention
}
```

- `TokenTable` keeps the pieces as one `string[]` and the scores as one `double[]` (Hugging Face's Unigram scores are
  64-bit), so a 250,000-piece vocabulary is two arrays, not a `JsonObject` of a quarter million nodes (rule 74).
- Guarded (Throw / FallBack / Shadow, compared by ids): an app's "Unigram" and the built-in can answer the same call.
- The built-in `"BPE"` builds a `BpeTokenizer` through its existing internal constructor path (`FromParts`).
- An unknown type: `Tokenizer model 'X' is not registered (registered: BPE, Unigram, WordPiece); add it with
  TokenizerModels.Register.` `BpeTokenizer`'s own message at `:94` gets the same hint.

**Files** (`src/Idrak/Models/Abstractions/TokenizerFiles.cs`): a registry of readers by name.

```csharp
public static class TokenizerFiles
{
    public static void Register(string name, Func<string folder, string? file> find,
        Func<string file, TokenizerOptions, TokenizerDefinition> read);
    // Unregister, Names, Find, Get, Default, Origin, Describe; unguarded: a reader's definition is its file's
}
```

L0 registers `"tokenizer.json"` (streams `model.vocab` and `model.merges` into the `TokenTable`, as `BpeTokenizer.Load`
streams them today). `"sentencepiece"` comes in T0, and `"vocab.txt"` in T2.

**Loader** (`src/Idrak/Models/Tokenizers/PretrainedTokenizers.cs`): `PretrainedTokenizers.Load(string folderOrFile,
TokenizerOptions? options = null)` finds the file (`Format`, or the first reader whose `find` answers), reads it and
creates its model. A folder with no reader's file: `No tokenizer file in X (tokenizer.json, tokenizer.model, vocab.txt,
or a registered reader's); add a reader with TokenizerFiles.Register.`

**Callers move** to `IPretrainedTokenizer` and `PretrainedTokenizers.Load`:
- `ModelFiles.Tokenizer`, `PretrainedModel.cs:381`
- `ITokenizerTrainer.Train`'s return type
- `BaseModelInfo`
- FineTuning, SpeechModels, SpeechSynthesis
- ServeHost, ModelFacts, the Create, Text, Data and Check commands
- the samples: Chat, IdrakAudio, WhisperSpeech
- the tests
- `PieceCacheLimit` replaces BPE's `CacheLimit` constant.

**Tests** (`tokenizers:`):
- the registry names BPE, refuses an unknown type with the hint, and an app's model registered and unregistered leaves
  it as found;
- every BPE fixture (`data/pretrain/*`, `data/gguf/*-hf`) loads through `PretrainedTokenizers.Load` to the ids
  `BpeTokenizer.Load` gave before (hash of all ids on `TokenizerSuite`'s texts, recorded before the change);
- `TokenizerModelSuite` passes for the built-in BPE.

Existing groups run unchanged: `pretrained:`, `tokenizer training:`, `plugins:`, `gguf:`.

**Done when:**
- no library file names `BpeTokenizer` outside its own and the registry's;
- the API dump and the inventory are regenerated;
- the BPE ids hash is unchanged.

**L0 as built (2026-10-19).** As planned, with these differences:
- `TokenizerDefinition` carries the tables as `Vocabulary` (`Dictionary<string, int>`, BPE and WordPiece),
  `Pieces` (`(string Piece, double Score)` list, Unigram) and `Merges`, in place of one `TokenTable`. BPE takes the
  dictionary as it was read, so loading copies nothing. A guarded registry gives the library's model its own copy, made
  only when an app overrides a built-in.
- File readers implement `ITokenizerFileReader` (`Name`, `Summary`, `Find`, `Read`), as the trainers implement
  `ITokenizerTrainer`, in place of two delegates.
- `PretrainedTokenizers` also has `TryLoad` (null when no reader finds a file; finds it once) and `Exists`, which the
  CLI, `PretrainedModel` and the serve host use in place of their own `File.Exists("tokenizer.json")` checks.
- The suite is `PretrainedTokenizerSuite` (`Conformance.CheckPretrained`): every `TokenizerSuite` check, plus ranges
  (the whole text and a slice, appended to a list that already holds ids), tokens found by name, `IsSpecial` false
  outside the vocabulary, and tokenizer.json naming the model.
- The baseline, `tests/Idrak.Tests/data/tokenizers/bpe-baseline.json`, was recorded at e53acb8 before any change. It
  covers all 23 `tokenizer.json` fixtures in the test data (888 texts: the suite's, two more, and `pretrain/corpus.txt`)
  and includes the refusals of tokenizers without an unknown token.
- The CLI's missing-tokenizer messages name the readers: "No tokenizer at X (readers: tokenizer.json)".
- Run (CPU, 4 threads): `tokenizers:` (6 new), `tokenizer training:`, `pretrained:`, `plugins:`, `gguf:`,
  `cli pretrain:`, `override loop:`, `conformance kit:`, `fine-tuning:`, `cli run:`, `cli models:`, `cli design:`,
  `cli measure:`, `cli serve:`, `cli train data:`, the audio train, seq2seq, tts train and cli groups, `public API` and
  `inventory`: 155 passed, 0 failed. Owner: no GPU run is needed for L0 (nothing on a device changed).

## Phase L1: the span pipeline with alignments

**Why.** Unigram and WordPiece need a normalizer and a pre-tokenizer that run without allocating, and offsets need every
character's origin. Both are missing from the string contracts.

**Types** (`src/Idrak/Models/Abstractions/`):
- `NormalizedText`: a pooled `char[]` plus an `int[]` origin per character (start and end in the original string), with
  `Replace`, `Prepend`, `Append`, `Remove`, `Transform` that keep the alignments as Hugging Face's `NormalizedString`
  does; it returns its buffers to the pool in `Dispose`.
- `PreTokenizedText`: the pieces as ranges of one `NormalizedText`, never substrings.

**Contracts gain a span form** (default interface methods, so registered plug-ins keep working):
- `ITokenizerNormalizer.Normalize(NormalizedText text)`: the default runs the string form and aligns the whole output to
  the whole input (offsets through such a component are coarse, and say so in its doc).
- `IPreTokenizer.PreTokenize(PreTokenizedText text, bool atStart)`: the default runs the string form and finds each
  piece in order in the normalized text.

**Pipeline** (`TokenizerPipeline.cs`): normalizer, then pre-tokenizer, then model, then post-processor, then decoder,
built once from the definition's `Pipeline`. Built-in components run on spans (`SpanComponents.cs`):
- normalizers: Sequence, NFC/NFKC/NFD/NFKD, Prepend, Replace, Lowercase, Strip;
- pre-tokenizers: Sequence, Split, Metaspace, Whitespace, WhitespaceSplit, Punctuation, Digits, ByteLevel;
- an app's component goes through its string form, as in BPE today.

Unicode normalization keeps its alignments by normalizing segment by segment between starters (combining class 0). It
uses .NET 10's span normalization APIs where the SDK has them, checked when building, else one string per segment.

**Post-processors**: a fourth `TokenizerComponents` table (`RegisterPostProcessor`, `PostProcessorTypes`, …), with
`IPostProcessor.Process(TokenizerEncoding single, TokenizerEncoding? pair, bool addSpecialTokens)`. The table is empty
in L1; T2 fills it.

**Tests** (`tokenizers: pipeline`):
- every built-in span component gives the string form's output on `TokenizerSuite`'s texts;
- alignments map every character back inside its origin;
- an app's string-only component still runs.

**Done when** the span pipeline gives the string pipeline's pieces for every BPE fixture's normalizer and pre-tokenizer.
BPE itself keeps its own paths (T5 decides whether it moves).

**L1 as built (2026-10-19).** As planned, with these differences and measurements:
- **Public types:**
  - `NormalizedText`: pooled characters, each with the range of the original text it stands for. A change is written
    through `Rewrite()` (`Append`, `AppendOrigin`, `Copy`, `Commit`), and `Set` takes a component's string result.
  - `PreTokenizedText`: the pieces as ranges of one `NormalizedText`.
  - `ITokenizerPostProcessor` and the fourth `TokenizerComponents` table, with no built-ins yet.
  - `TokenizerEncoding`, in `Idrak.Abstraction`, which T3 builds on.
- **Span forms:** `ITokenizerNormalizer.Normalize(NormalizedText)` and `IPreTokenizer.PreTokenize(PreTokenizedText, bool)`
  are default methods, so string-only plug-ins keep working.
  - A string-only normalizer's change stands for the changed middle as a whole.
  - A string-only pre-tokenizer's pieces keep their ranges where they are found in order; a rewritten piece stands for
    the piece it came from.
- **The built-ins on spans** (`SpanComponents`, internal; plain delegates, so there are no new abstract types):
  - normalizers: Sequence, NFC/NFKC/NFD/NFKD, Prepend, Replace (literal, regex, `$` substitutions), Lowercase, and the
    new **Strip**;
  - pre-tokenizers: Sequence, Split, ByteLevel, Metaspace, Digits, Whitespace, and the new **WhitespaceSplit** and
    **Punctuation**. The three new types also have string forms in `BpeTokenizer`, so BPE files that use them load.
  - `TokenizerPipeline` (internal) runs a tokenizer.json's normalizer and pre-tokenizer on a range of a text.
- **Character ranges:** a replaced space stands for the space; a prefix for the piece's first character; a byte
  character for the character it is a byte of; an NFKC cluster for its characters. T3 checks these against the
  references' offsets and adjusts them where Hugging Face differs.
- **Unicode normalization without the platform where possible:** .NET's span normalization (`IsNormalized`,
  `GetNormalizedLength`, `TryNormalize`) allocates 72 bytes on every call on Linux (ICU), measured.
  - A per-form table of the characters no context can change (their own normalization is themselves, not a mark, not a
    Hangul vowel or final jamo) is computed once per block of 256 code points.
  - Text made only of such characters is left as it is with no platform call: ASCII, Arabic without diacritics, CJK,
    most emoji.
  - Text normalization does change costs 624 bytes a text on the test's samples (Arabic with harakat, a ligature,
    fullwidth letters).
  - **Candidate (CLAUDE.md, the reference is not the ceiling):** normalization in C# from tables built once, with no
    platform calls, would remove those bytes too.
- **Proven equal:** every built-in span component gives the pieces its string form gives. This was checked over every
  `tokenizer.json` in the test data and 31 hand-made specs, on the suite's texts and both corpora, at the start and
  not. Specs whose types later phases build (Precompiled, Bert*) are skipped and counted.
- **Allocation:** a warm run allocates 0 bytes on Latin, Arabic, CJK and emoji text through NFKC + Replace + Lowercase
  + Strip + Metaspace, through Split + ByteLevel, and through Replace + WhitespaceSplit + Punctuation + Digits.
- **Run (CPU):** `tokenizers:`, `plugins:`, `pretrained:`, `tokenizer training:`, `override loop:`, `gguf:`,
  `public API`, `inventory`.
  - First run: 2 failures. One was the inventory: the two abstract classes became delegates. The other was the
    allocation test: the stability table was added.
  - After one rebuild, the failing filters and the pipeline group pass, and so do `plugins:`, `override loop:` and
    `inventory`.

## Phase T0: fixtures, the SentencePiece reader, the Precompiled normalizer

**Fixtures** (`tools/tokenizers/make_fixtures.py`, run with `python -I`; `tokenizers`, `sentencepiece` and `transformers`
pinned in `tools/tokenizers/requirements.txt`, versions in the manifest):
- **Corpus** (`data/tokenizers/corpus.txt`): Arabic with diacritics, tatweel and mixed RTL/LTR with numbers, emoji with
  skin tones and flags, CJK, accents, combining marks, whitespace runs, tabs and CRLF, empty lines, very long words,
  control characters.
- **Trained small tokenizers**, about 50-200 KB each, from the corpus plus a public-domain text:
  - SentencePiece Unigram, with and without byte fallback;
  - SentencePiece BPE;
  - Hugging Face WordPiece, uncased and cased;
  - each as `tokenizer.model` (SentencePiece's own) and as `tokenizer.json` (converted with `transformers`'
    `convert_slow_tokenizer`, the way Hub files are made).
- **References**: per tokenizer and text, the ids, offsets (UTF-16), type ids, special-tokens mask and decoded text. For
  `.model` files they come from `sentencepiece`, for `.json` from `tokenizers`. Where the two libraries disagree on the
  same model, the cases are listed in `known-differences.json`, not hidden.
- **`--real`** (the owner): downloads with `huggingface_hub` into the cache (`IDRAK_CACHE`) and writes only the
  references to `data/tokenizers/real/<name>.json`. The models:
  - `google-t5/t5-small` and `google/mt5-small` (Unigram)
  - `albert/albert-base-v2` (Unigram)
  - `FacebookAI/xlm-roberta-base` (Unigram, Arabic)
  - `hf-internal-testing/llama-tokenizer` (SentencePiece BPE)
  - `google-bert/bert-base-uncased` and `google-bert/bert-base-multilingual-cased` (WordPiece)
  - `aubmindlab/bert-base-arabertv02` (WordPiece, Arabic)

  The list is the owner's to change.

**Protobuf reader** (`ProtoReader.cs`): a `ref struct` over `ReadOnlySpan<byte>` (varint, fixed32, fixed64,
length-delimited; unknown fields skipped), shared with the ONNX reader if it fits without changing ONNX results.

**SentencePiece model** (`SentencePieceModel.cs`): reads `ModelProto`, with the field numbers from
`sentencepiece_model.proto` at the pinned version:
- the pieces, with score and type (normal, unknown, control, user-defined, byte, unused);
- the trainer spec: model type, byte fallback, split digits, treat whitespace as suffix, unk/bos/eos/pad ids and pieces;
- the normalizer spec: rule name, precompiled charsmap, add dummy prefix, remove extra whitespaces, escape whitespaces.

The `"sentencepiece"` reader (`tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model`) turns it into a
`TokenizerDefinition`:
- a Unigram model → `"Unigram"`;
- a BPE model → `"BPE"`, with merges derived from the scores as Hugging Face's `SentencePieceExtractor` does (so there is
  one BPE implementation);
- the normalizer spec → Precompiled (plus Replace for extra whitespace) and Metaspace with the dummy-prefix scheme, as
  `convert_slow_tokenizer` writes them.

**Double-array trie** (`DoubleArrayTrie.cs`): reads darts-clone's unit array (common-prefix search over UTF-8 bytes) and
builds one from sorted keys. The same type serves the charsmap (T0) and the piece tries (T1, T2): one implementation.

**Precompiled normalizer** (`PrecompiledNormalizer.cs`, registered as `"Precompiled"`): the charsmap blob (trie size,
trie units, then the normalized strings, NUL-terminated) is read once into the trie plus one byte array. It replaces as
`tokenizers`' `Precompiled` does (by grapheme cluster, falling back to characters), keeping alignments.

**Tests** (`tokenizers: sentencepiece`):
- each fixture's `ModelProto` reads to `sentencepiece`'s piece table (pieces, scores, types) and flags;
- a truncated or corrupt file fails naming the offset;
- Precompiled normalizes the corpus exactly as `tokenizers` does.

**T0, wave 1, as built (2026-10-19).** Fixtures, `ProtoReader`, `SentencePieceModel` and `DoubleArrayTrie` (all
internal), and the groups `tokenizers: sentencepiece` and `tokenizers: trie`: 6 tests, 17 with L0's, `public API` and
`inventory` after the merge, all passing on the CPU. The fixtures' README (`tests/Idrak.Tests/data/tokenizers/README.md`)
lists every file. What the plan did not know:

- **Versions:** tokenizers 0.23.2, sentencepiece 0.2.2, transformers 5.18.0, protobuf 7.36.2, huggingface_hub 1.33.0;
  Python's Unicode data 15.1.0. Three runs of the script wrote the same bytes. The fixtures take 2.6 MB in all.
- **Field numbers** (sentencepiece 0.2.2):
  - `ModelProto`: pieces 1, trainer_spec 2, normalizer_spec 3.
  - `SentencePiece`: piece 1, score 2 (a float), type 3. The types are normal 1, unknown 2, control 3, user-defined 4,
    unused 5, byte 6.
  - `TrainerSpec`: byte_fallback 35, split_digits 25, treat_whitespace_as_suffix 24, unk/bos/eos/pad ids 40-43
    (defaults 0, 1, 2, -1).
  - `NormalizerSpec`: name 1, precompiled_charsmap 2, add_dummy_prefix 3, remove_extra_whitespaces 4,
    escape_whitespaces 5.
- **The charsmap** is laid out as planned. Its trie is a DAWG with shared suffixes: `nmt_nfkc` holds 225,275 keys in
  44,800 units, and the C# enumeration equals an independent Python reader's.
- **Merges from a SentencePiece BPE model:** transformers 5's converter for this is broken. T0 (the reader) and T3
  (GGUF `llama`) derive the merges as transformers 4's `generate_merges` does: sorted by score, then the left piece's
  length, then the right one's, all descending, with a stable sort.
- **The two references disagree on 21 corpus lines** (`known-differences.json`):
  - `<unk>` decodes as " ⁇ " in sentencepiece, and Hugging Face skips it.
  - Viterbi breaks ties differently in a run of 300 `a`s.
  - Hugging Face matches special-token text written in the input; sentencepiece does not.
  - Converted files strip trailing tabs and CRs.
  - Hugging Face's `Precompiled` replaces a whole grapheme by its first prefix key, which drops the following mark (أَ
    becomes ا); sentencepiece keeps the mark.

  Each path is held to its own reference, as the Risks section says. T1's Precompiled follows `tokenizers` for
  tokenizer.json and sentencepiece for tokenizer.model.
- **Byte fallback offsets:** sentencepiece gives every byte piece of one character that character's offsets, and every
  piece but the last is empty (begin equals end).
- **WordPiece training is not repeatable in `tokenizers`:** its `WordPieceTrainer` numbers the `##` alphabet in hash
  order, and two processes learned 39 different pieces. The fixtures' WordPiece vocabularies come from a deterministic
  SentencePiece BPE run over BERT-normalized words; the tokenizers and their references are still `tokenizers`' own.
  **T4 changes accordingly:** the `"frequency"` WordPiece trainer cannot be checked against Hugging Face's vocabulary
  exactly. It is checked against the same algorithm with ties broken in a fixed order (written into the script), and
  against Hugging Face's by overlap above a measured threshold.
- **Post-processors:** `BertConverter` writes `TemplateProcessing`, so `wordpiece-uncased` uses `TemplateProcessing` and
  `wordpiece-cased` uses `BertProcessing`; T2 covers both.
- **The owner's `--real` run** (from `D:\Projects\Idrak`):
  ```powershell
  python -m venv $env:USERPROFILE\tok
  & $env:USERPROFILE\tok\Scripts\python.exe -m pip install -r tools\tokenizers\requirements.txt
  & $env:USERPROFILE\tok\Scripts\python.exe -I tools\tokenizers\make_fixtures.py --real
  ```
  It downloads into `IDRAK_CACHE\tokenizers\<org>--<name>\`, or `~/.cache/idrak` when `IDRAK_CACHE` is unset, and
  writes only `tests\Idrak.Tests\data\tokenizers\real\<name>.json`.

**T0, the rest, as built (2026-10-19)** (agent B, merged with L1, then with T2):
- **The "sentencepiece" reader** (`TokenizerFiles.SentencePiece`, asked after tokenizer.json and before vocab.txt):
  - Unigram and BPE models are read; WORD and CHAR models fail naming the type.
  - BPE merges come by the `generate_merges` rule.
  - The definition writes sentencepiece's normalizer as standard steps: Precompiled in sentencepiece's mode, Replace
    `\A +`→"", Replace `" {2,}"`→" ", Prepend ▁, Replace " "→▁, Replace `(?:▁)+\z`→"", and a suffix rule for
    `treat_whitespace_as_suffix` (implemented, but no fixture covers it).
  - The pre-tokenizer is `Metaspace` (prepend never, split) only when no piece holds ▁ past its first character.
  - A leading `Replace <unk>→" ⁇ "` in the decoder gives sentencepiece's unknown text.
  - No post-processor, since sentencepiece adds no BOS or EOS.
- **Evidence:** the C#-made definitions, saved as tokenizer.json and run through Python `tokenizers`, give
  sentencepiece's ids and text on every line not in `known-differences.json`: unigram 99/102, unigram-bytes 97/102,
  bpe 100/102.
- **Precompiled** (`PrecompiledCharsMap`, string and span forms):
  - Two modes: Hugging Face's (by grapheme) and sentencepiece's, chosen by `"mode": "sentencepiece"` on the normalizer.
    `tokenizers.Tokenizer.from_str` accepts the extra field and drops it when it saves, so files stay loadable.
  - Both modes equal `normalized.json` on every line.
  - There were no .NET-vs-Rust grapheme differences on the corpus.
  - Text that does not change comes back as the same string.
- **Other changes:**
  - `Prepend` leaves an empty text empty, as `tokenizers` does (string and span).
  - `DoubleArrayTrie` gained `ShortestPrefix` and `LongestPrefix`.
- **GGUF:**
  - uint8 and int8 arrays are read as `byte[]` and `sbyte[]`, float32 as `float[]`, with `Get<long[]>` and
    `Get<double[]>` converting.
  - A real bug was found and fixed: the runtime treats `byte[]` and `sbyte[]` alike, so int8 arrays were read as
    unsigned.
  - Numeric arrays are read in one block. `FormatVersion` is unchanged: prepared files did not change.
- **After both merges:** one build, `tokenizers:`, `gguf:`, `plugins:`, `pretrained:`, `tokenizer training:`,
  `override loop:`, `public API` and `inventory`: 76 passed, 0 failed.

## Phase T1: Unigram

**Model** (`UnigramTokenizer.cs`, registered as `"Unigram"`), from `model.vocab` (pieces and scores), `unk_id` and
`byte_fallback`:

- **Viterbi** over a byte lattice of the normalized, pre-tokenized piece in UTF-8 (converted once into a pooled buffer).
  Nodes start only at character boundaries. The piece trie gives every match from a position in one walk. Best scores
  are summed in a fixed left-to-right order in `double`, and ties break as the reference does.
- **Unknown handling**: unknown characters take the unknown piece's score with Hugging Face's penalty, and runs of
  unknowns fuse as `fuse_unk`. With byte fallback, an unknown character becomes its `<0xNN>` pieces.
- User-defined and control pieces are added tokens, matched first, as BPE matches them.
- Word results are cached by span (`PieceCacheLimit`), as BPE caches its pieces.
- **Sampling** (off by default): `SamplingAlpha` and `SamplingBest`. N-best sampling as SentencePiece's `SampleEncode`,
  or forward-filtering backward-sampling when the n-best size is 0 or negative, seeded by `SamplingSeed`. A sampled
  encode bypasses the cache.
- **Decode**: the pipeline's decoders (Metaspace, ByteFallback, Fuse, Strip, Replace) as BPE runs them; byte pieces are
  reassembled to UTF-8 (invalid sequences become U+FFFD, as the reference does).

**Tests** (`tokenizers: unigram`):
- the ids equal the reference for every Unigram fixture and corpus line, from `tokenizer.json` and from
  `tokenizer.model`;
- decode(encode(x)) equals the reference's decode;
- byte fallback round-trips arbitrary bytes;
- sampling with a seed is repeatable, and alpha null gives the Viterbi ids;
- `EncodeRange` into a reused list allocates 0 bytes once warm;
- `TokenizerSuite` and `TokenizerModelSuite` pass.

**T1 as built (2026-10-19).** `UnigramTokenizer` (internal, `src/Idrak/Models/Tokenizers/UnigramTokenizer.cs`),
registered as `"Unigram"` beside BPE (`TokenizerModels.Unigram`, public). Group `tokenizers: unigram` (7 tests,
`tests/Idrak.Tests/TokenizerUnigramTests.cs`). What the plan did not know:

- **The 300-`a` tie is float rounding, not a tie rule.** Both references keep the first of equal scores; sentencepiece
  sums in `float`, tokenizers in `double`. So the model has a `mode` field, as Precompiled has: the sentencepiece reader
  writes `"mode": "sentencepiece"` on the Unigram model (scores summed in `float`; control, unknown and byte pieces not
  matched in text; user-defined pieces with sentencepiece's bonus; unknown score = lowest *normal* score − 10), and
  without it the lattice is tokenizers' (`double`, every piece in the trie, unknown score = lowest score − 10).
  tokenizers ignores the field, so a saved file still loads there. Line 51 now agrees on both paths.
- **Agreement** (ids, then ids and decoded text, of 102 corpus lines):
  - tokenizer.json vs tokenizers: `unigram` 102 / 94, `unigram-bytes` 102 / 100. Every text difference is a known
    line: tokenizers' `decoded` skips special tokens, `Decode` does not (`<unk>` stays `<unk>`, `</s>` stays). The
    reference's ids with the post-processor's tokens decode to its `decoded_with_special` on every line.
  - tokenizer.model vs sentencepiece: 100 / 100 on both. Lines 59 and 60 differ: the reader makes control pieces added
    tokens, so `</s>`, `<unk>`, `<cls>` written in text are matched; sentencepiece never matches control pieces.
    Candidate: a per-added-token "not matched in text" flag for the .model path.
- **Unknowns and bytes:** a run of unknown characters fuses into one unknown id; with byte fallback it becomes the
  `<0xNN>` pieces of its bytes (the unknown id if one is missing, as tokenizers). Arbitrary characters round-trip through
  byte fallback on both paths.
- **Metaspace on empty text:** text that normalizes to nothing (`\r`, blanks, a run of tatweel under the user rule)
  gave a lone `▁`; tokenizers gives nothing (`NormalizedString::prepend` does nothing on empty text). Fixed in all three
  forms (span, string stage, BPE's fast path); the BPE baseline is unchanged.
- **The suite's "round trip is stable"** cannot hold where text decodes to the unknown piece's text: `<unk>` is an added
  token and `" ⁇ "` normalizes to `??`, and both re-encode with `▁` after them, in both references. The test names
  those cases (the `unigram` fixture, and `unigram-bytes` from the .model) and fails on anything else.
- **Sampling:** `SamplingAlpha`, `SamplingBest` (default -1: forward filtering, backward sampling over the whole
  lattice; more than 1: n best by a k-best pass, drawn by exp(alpha × score); 1: no sampling), `SamplingSeed`, with
  `IDRAK_TOKENIZER_ALPHA`, `_NBEST`, `_SEED`. One generator per tokenizer, locked; a sampled encode bypasses the cache.
  The `unigram-bytes` model's scores are so far apart that sentencepiece itself samples 2 of 60 lines other than the
  best at alpha 0.1 (ours: 2), so the test samples the `unigram` fixture (sentencepiece: 48 of 60).
- **Memory:** a thread keeps its lattice arrays for pieces up to 64 KiB of UTF-8 (longer pieces rent and return them);
  `EncodeRange` into a reused list allocates 0 bytes once warm on every corpus line, both fixtures, both paths, with
  the piece cache and without it.
- **Taken from measurements (the reference is not the ceiling):**
  - `ConcurrentDictionary.Count` takes every lock; the cache size check paid it on each miss (and on every piece with
    the cache off). Unigram and BPE now count on add. No-cache Unigram went from 6.2M to 8.7-9.3M tokens/s on one
    thread, from 10.6M to 25-27M on four.
  - The lattice's arrays came from the pool six times a piece; now a thread keeps them.
  - Precompiled (tokenizers' mode) looked up every ASCII character through a grapheme split, an encode and a trie walk;
    an ASCII character that is its own cluster (followed by ASCII, not CR LF) now takes a 128-entry table. The
    tokenizer.json path went from 14.7M to 18.4M tokens/s warm.
- **Speed** (the corpus' non-blank lines repeated to 200,640 UTF-16 characters, as `--bench` repeats them to 204,136
  code points; tokens a second, a 4-thread Linux container, noisy to about ±20%):

  | | one thread cold | one thread warm | 4 threads warm |
  |---|---|---|---|
  | sentencepiece (`unigram`) | 6.84M | 7.45M | 19.0M |
  | tokenizers (`unigram`) | 1.01M | 1.14M | 2.05M |
  | Idrak, tokenizer.json, cache | 8.5M | 18.4M | 49.1M |
  | Idrak, tokenizer.model, cache | 3.2M | 18.4M | 34.9M |
  | Idrak, tokenizer.json, no cache | 6.2M | 9.4M | 29.5M |
  | Idrak, tokenizer.model, no cache | 6.5M | 10.6M | 26.7M |
  | sentencepiece (`unigram-bytes`) | 9.0M | 11.7M | 20.6M |
  | Idrak (`unigram-bytes`), json / model, cache | 12.1M / 10.9M | 36.2M / 37.5M | 89.7M / 64.9M |

  Measured in the test `tokenizers: unigram: speed` (the four Idrak rows per fixture), not yet in `--bench-text`.
- **Candidates left:** the normalizer and pre-tokenizer are 38-64% of a no-cache pass (the `Replace` regexes, Strip,
  sentencepiece-mode Precompiled encoding the whole text); a cold first pass on one thread runs tier-0 JIT code (the
  .model path's 3.2M); BPE's `DecoderStage` Metaspace strips one leading space whatever the prepend scheme, where
  tokenizers drops every `▁` of the first token unless the scheme is `never` (the same text on every fixture line; T5
  can align it). The reader reads unused pieces as normal ones (sentencepiece skips them; no fixture has one). Offsets
  are T3's.

## Phase T2: WordPiece

**Components** (`BertComponents.cs`), as `tokenizers` defines them, each keeping alignments:
- `BertNormalizer`: `clean_text`, `handle_chinese_chars`, `strip_accents` (null follows `lowercase`), `lowercase`;
- `BertPreTokenizer`: whitespace and punctuation splits;
- the `WordPiece` decoder (`prefix`, `cleanup`);
- registered in `TokenizerComponents`, so `ModelPluginTests.cs:71` moves from 7 decoder types to 8.

`Punctuation`, `WhitespaceSplit` and `Strip` were added in L1, and the `Prepend` decoder is added here.

**Model** (`WordPieceTokenizer.cs`, registered as `"WordPiece"`), from `vocab`, `unk_token`,
`continuing_subword_prefix` and `max_input_chars_per_word`:
- greedy longest match first, using two tries: word-initial pieces, and continuation pieces stored without their
  prefix. Each step is one walk, with no `"##" + piece` string built.
- A word longer than `MaxInputCharsPerWord` becomes `[UNK]`.

**Post-processors** (`PostProcessors.cs`): `TemplateProcessing` (single and pair templates, special token ids, type
ids), `BertProcessing`, `RobertaProcessing` (with `trim_offsets` and `add_prefix_space`) and `Sequence`.

**`vocab.txt` reader** (registered as `"vocab.txt"`): one token a line, with `tokenizer_config.json` (`do_lower_case`,
`strip_accents`, `tokenize_chinese_chars`, the special tokens), turned into the definition `BertTokenizerFast` would
write.

**Tests** (`tokenizers: wordpiece`):
- the ids equal the reference for every WordPiece fixture, from `tokenizer.json` and from `vocab.txt`;
- the decode equals the reference's;
- a long word becomes `[UNK]`, and `MaxInputCharsPerWord` changes the limit;
- `[CLS] … [SEP]` and pair type ids equal the reference;
- `plugins:` passes with the new count.

**T2 as built (2026-10-19).** As planned, with these differences and findings:
- **BERT components** (`BertComponents`), each with one span implementation that the string form runs:
  - `BertNormalizer`, matching `tokenizers`, which the probe in the reference venv showed:
    - `clean_text` drops U+0000, U+FFFD and categories Cc, Cf and Co, but not Cn: an unassigned U+0378 is kept;
    - whitespace becomes a space;
    - CJK ideographs get a space on each side, both spaces standing for the ideograph;
    - `strip_accents` is NFD without the Mn marks, and when null it follows `lowercase`.
  - `BertPreTokenizer`: WhitespaceSplit, then Punctuation (Isolated).
  - The `WordPiece` decoder: the `##` prefix is stripped from tokens after the first, others get a space, and
    `tokenizers`' cleanup runs on each token.
- **No `Prepend` decoder:** the plan named one, but `tokenizers` has no such decoder type, so none was added.
- **Lowercase bug fixed:** `tokenizers` lowercases with Rust's `char::to_lowercase`, which turns İ (U+0130) into i
  plus U+0307. .NET's invariant mapping leaves İ unchanged, so `Lowercase` (string and span) and `BertNormalizer` now
  map İ as Rust does. The L0 BPE baseline is unchanged.
- **The `WordPiece` model** (`WordPieceTokenizer`, registered as `TokenizerModels.WordPiece`):
  - Two `DoubleArrayTrie`s: word-initial entries, and continuation entries without their prefix. The longest match is
    the last entry of one common-prefix walk, so there is no `"##" + piece` string and no allocation.
  - Word ids are cached by span (`PieceCacheLimit`).
  - `MaxInputCharsPerWord` / `IDRAK_TOKENIZER_MAX_WORD_CHARS`.
  - Added tokens are matched first, as BPE matches them. `normalized`, `lstrip`, `rstrip` and `single_word` are not
    honoured yet, as in BPE; T3 adds them where the references need it.
- **Post-processors** (`PostProcessors`, the table's built-ins): `TemplateProcessing`, `BertProcessing`,
  `RobertaProcessing` (pair type ids 1, as `tokenizers` sets them), `ByteLevel` (no tokens; T3 trims its offsets) and
  `Sequence` (offset-only processors, then at most one that adds tokens).
- **The `vocab.txt` reader** (`TokenizerFiles.VocabTxt`): it reads `vocab.txt` and `tokenizer_config.json` into the
  definition `BertConverter` writes. A byte-order mark stays in the first token, as transformers reads it, and a token
  that repeats takes its later id.
- **Results (CPU):** both fixtures' `tokenizer.json` match `tokenizers` on all 102 corpus lines and on the pairs:
  - ids with the post-processor's special tokens, type ids, special-tokens mask and sequence ids;
  - decoded text with special tokens, and without them (special ids left out, as `skip_special_tokens` does);
  - `vocab.txt` gives the same ids and special tokens on every line;
  - the pretrained tokenizer suite passes, a saved tokenizer loads back to the same ids, and a warm encode allocates 0
    bytes on Latin, Arabic, CJK and upper-case text.
  - Text with accents to strip goes through NFD, and so through the platform's normalization (L1's candidate).
- **Run:** `tokenizers:`, `plugins:`, `pretrained:`, `tokenizer training:`, `override loop:`, `public API`, `inventory`
  gave 66 passed and 1 failed. The failure was a test comparing with the reference's `decoded`, which skips special
  tokens (`[UNK]` among them); the test was fixed and its filter reran: 5 passed.

## Phase T3: encodings, detection, GGUF

**Encoding** (`src/Idrak.Abstraction/Generation/Tokenizer.cs`):

```csharp
public sealed record TokenizerEncodeOptions
{
    public bool AddSpecialTokens { get; init; }             // false: as Encode today (chat templates write them)
    public int? MaxLength { get; init; }
    public string Truncation { get; init; } = "longest_first";   // also "only_first", "only_second"
    public string Padding { get; init; } = "none";          // also "longest", "max_length"
}

public sealed class TokenizerEncoding
{
    public int[] Ids, TypeIds, AttentionMask, SpecialTokensMask;
    public (int Start, int End)[] Offsets;                  // UTF-16 indexes into the text (or the pair's text)
    public int[] SequenceIds;                               // 0, 1, or -1 for special tokens
}

// on IPretrainedTokenizer:
TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options);
IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs,
    TokenizerEncodeOptions options);
string Decode(ReadOnlySpan<int> ids, bool skipSpecialTokens);
```

- Unigram and WordPiece take offsets from the span pipeline.
- BPE gets an offsets path that runs only when offsets are asked for; its `Encode` fast path does not change.
- Batch encoding uses `Workers` threads, one text per item, and returns results in input order.
- `tokenizer.json`'s `truncation` and `padding` are read as the encode options' defaults.

**Detection**: `PretrainedTokenizers.Load` covers `tokenizer.json`, `tokenizer.model`, and `vocab.txt` with
`tokenizer_config.json`. A file whose model type nobody registered fails with `TokenizerModels`' hint.

**GGUF** (`GgufModel.WriteTokenizer`):
- `llama` → `"BPE"` with merges from the scores and byte fallback, as the SentencePiece reader does;
- `bert` → `"WordPiece"` with `BertNormalizer`, `BertPreTokenizer` and `BertProcessing`;
- `t5` → `"Unigram"` with Precompiled from `tokenizer.ggml.precompiled_charsmap`;
- `gpt2` stays as it is;
- any other model fails naming it, with a `TokenizerModels.Register` hint.

The reader also changes: `Gguf.cs` returns uint8 and int8 arrays as `byte[]` and `sbyte[]`, and float32 arrays as
`float[]`, with `Get<long[]>` and `Get<double[]>` converting for callers that ask for the old types. `FormatVersion`
(`GgufModel.cs:52`) goes from 1 to 2, so prepared folders are made again.

`tools/gguf/make_fixtures.py` gains tiny `llama`-, `bert`- and `t5`-tokenizer GGUFs made from T0's trained tokenizers.

**T3, GGUF, as built (2026-10-19).**
- **A registry, `GgufTokenizers`** (decision 10), by `tokenizer.ggml.model`. It replaces the fixed `gpt2` check, and an
  unknown model fails with "not registered … GgufTokenizers.Register". Built-ins (`GgufTokenizerFormats`):
  - **`gpt2`:** the old code moved as it was.
  - **`llama` and `t5`:** the GGUF's pieces, scores, token types, special ids, `add_space_prefix`,
    `remove_extra_whitespaces` (default false for llama, true for t5) and `precompiled_charsmap` are written back as the
    SentencePiece `ModelProto` they came from (`SentencePieceWriter`, which T4 extends). That is read as a
    tokenizer.model is, so a GGUF and its tokenizer.model give one tokenizer.
  - **`bert`:** llama.cpp's "phantom space" pieces are turned back (▁x → x, x → ##x, [X] kept), with the special tokens
    by their ids, through the definition the vocab.txt reader writes (`VocabTxtReader.BertDefinition`, shared).
    GGUF does not record lowercasing; it lowercases and strips accents, as llama.cpp's tokenizer does, and says so in
    a note.
- **Prepared tokenizer.json:** written from the definition (`TokenizerJsonReader.Write`). `FormatVersion` 2, so cached
  prepared folders are made again.
- **Tests** (`tokenizers: gguf`, GGUF files written by the test):
  - a llama GGUF made from the bpe fixture writes the tokenizer.json its tokenizer.model reads to, byte for byte, and
    encodes and decodes every corpus line as it;
  - t5 GGUFs made from unigram and unigram-bytes write identical tokenizer.json files;
  - a bert GGUF in llama.cpp's form gives tokenizer.json's ids on every line;
  - an unknown tokenizer fails, and an app's converter is used, then unregistered.
- **Run (CPU):** `tokenizers:`, `gguf:`, `plugins:`, `pretrained:`, `public API`, `inventory`: 66 passed, 0 failed.
- **Owner:** a real Llama-2-style GGUF (`tokenizer.ggml.model` llama) end to end, see "The owner's runs".

**Callers**: `TextEncoder` (Idrak.Nlp) pads and masks through `EncodeBatch` when its tokenizer is an
`IPretrainedTokenizer` (the same vectors, one tokenize pass); `idrak tokenize` gains `--offsets`, `--pair TEXT` and
`--special`.

**Tests** (`tokenizers: encoding`, `tokenizers: gguf`):
- offsets, type ids and masks equal the reference for every fixture (BPE included);
- truncation and padding equal the reference;
- batch encoding equals single encoding for 1 and many workers;
- each GGUF fixture gives the ids of the Hugging Face files it was made from;
- `gguf:` passes with `FormatVersion` 2.

**T3, encodings, as built (2026-10-19).** As planned, with these differences and findings:
- **Contract:** `IPretrainedTokenizer.Encode(text, pair, options)`, `EncodeBatch(texts, pairs, options)` and a default
  `Decode(ids, skipSpecialTokens)`. `TokenizerEncodeOptions` is `AddSpecialTokens` (default true, as `tokenizers`),
  `MaxLength`, `Truncation`, `Padding` and `PadId`; a null option takes the tokenizer.json's `truncation` and `padding`.
  `TokenizerEncoding` gains `Tokens`. The plain `Encode(string)` path does not change.
- **Shared steps** (`TokenizerEncodings`, internal): settings, truncation, the post-processor, padding, batches over
  `Workers` (`IDRAK_TOKENIZER_WORKERS`, results in input order), and the alignment of a piece's tokens to its characters
  (matching text; byte-fallback tokens of one character all span it; an unknown token spans up to the next token's
  text, or one character).
- **tokenizers 0.23's early exit, followed:** with truncation on, each sequence is encoded a pre-token at a time until
  it has `MaxLength` tokens (from its end for left truncation; the first text under `only_second` whole), and
  `longest_first` compares those lengths, not the full ones. Its own `truncate_encodings` alone gives other splits:
  32 and 42 tokens to 13 give 6/7 by the formula, 7/6 through `encode` (both cut to about 16 first, and the first is
  longer). Idrak does the same, which also stops encoding a long text early.
- **Normalization offsets as `tokenizers` aligns them:** a changed cluster is aligned by position, as
  `unicode_normalization_alignments` and `NormalizedString.transform` do: each source character decomposes alone (its
  first character kept, the rest added), canonical reordering moves them, a character composed of k parts removes
  k - 1; a kept character stands for the next source character, an added one for the one before. So "e" with a
  combining acute composed stands for the "e" alone, and a reordered mark for the source position, not its own
  character. Checked per character for NFC, NFKC, NFD and NFKD (pipeline test).
- **Post-processors:** `RobertaProcessing` and `ByteLevel` trim offsets (`trim_offsets`, `tokenizers`'
  `process_offsets`); a `Sequence` applies ByteLevel's trim first.
- **A tokenizer without an unknown token** refuses a character its vocabulary lacks (as `Encode` does since before
  plan 24), where `tokenizers` drops it: such corpus lines are counted, not compared (`pretrain/metaspace`: 87 of 103).
- **References:** `tools/tokenizers/make_encodings.py` writes `data/tokenizers/encodings.json` (tokenizers 0.23.2):
  seven BPE tokenizers on every corpus line (two derived from `pretrain/tokenizer` with the ByteLevel and
  RobertaProcessing post-processors), WordPiece truncation by each strategy, and padding to the longest and to 24.
- **Run (CPU):** `tokenizers:`, `abstraction inventory`, `public API`: 42 passed, 0 failed.
- **With T1 merged:** Unigram has the same detailed encode (its pieces through the piece cache, then aligned), checked
  against `unigram` and `unigram-bytes`' expected-hf.json (lines and pairs). Three more of `tokenizers`' alignment rules
  are followed:
  - `Replace`'s content stands for the occurrence's last character (`NormalizedString.replace` inserts it after the
    removed ones), so " {2,}" → "▁" stands for the last space.
  - `Precompiled` (Hugging Face mode) aligns by position, through the same cursor as `transform`: the i-th character of
    a replacement stands for the i-th it replaces, extra ones for the last, and characters it drops move the cursor on,
    except at the text's start, where `tokenizers` has no earlier character to charge them to and everything after
    stands one place earlier per dropped character. Idrak reproduces that too (the reference is the reference); the
    sentencepiece mode keeps whole-replacement ranges.
  - Unigram fuses a run of unknown characters before byte fallback, so each byte token spans the whole run.
  - A raw sequence's type id is its number (0, or 1 for the pair), as `tokenizers` encodes them; without a
    post-processor the pair keeps type id 1.
- **Run (CPU), with T1:** `tokenizers:`, `tokenizer training`, `pretrained:`, `gguf:`, `plugins:`,
  `abstraction inventory`, `public API`: 84 passed, 0 failed.

## Phase T4: trainers

**Contract**: `TokenizerTrainingOptions` gains:

| Option | Default | Meaning |
|---|---|---|
| `CharacterCoverage` | 0.9995 | share of characters kept in the alphabet |
| `ByteFallback` | false | add `<0xNN>` pieces |
| `MaxPieceLength` | 16 | longest piece |
| `SeedSize` | 1,000,000 | seed vocabulary before pruning |
| `ShrinkingFactor` | 0.75 | share kept per pruning round |
| `EmIterations` | 2 | EM steps per round |
| `SplitDigits` | false | split numbers into digits |
| `Lowercase`, `StripAccents` | false | WordPiece |
| `Scoring` | `"frequency"` | WordPiece: `"frequency"` or `"likelihood"` |
| `Arabic` | none | `"tatweel"`, `"diacritics"`, or both |

The Arabic choices are written into the tokenizer as standard `Replace` normalizers, so Hugging Face can read the file
too. `TokenizerTrainingProgress` gains the stages `"seeds"`, `"em"` and `"prune"`.

**Unigram trainer** (`UnigramTrainer.cs`, registered as `"unigram"`), SentencePiece's procedure:
- Seed pieces are frequent substrings, found through a suffix array (SA-IS, in C#) of the corpus.
- Then EM rounds: an E step of forward-backward over each sentence's lattice, an M step on the expected counts (with
  SentencePiece's digamma correction), then pruning by loss to `ShrinkingFactor`, until `VocabularySize`.
- Required characters are added by `CharacterCoverage`.
- The E step is spread over `Workers`, with per-chunk sums combined in chunk order, so one worker and many give the same
  vocabulary.
- It writes `tokenizer.json`, and `tokenizer.model` through `SentencePieceWriter.cs` (the T0 fields, written back).

**WordPiece trainer** (`WordPieceTrainer.cs`, registered as `"wordpiece"`):
- `"frequency"` follows Hugging Face's `WordPieceTrainer` (BPE merges with the `##` prefix), so its vocabulary is
  checked exactly, as the BPE trainer's is.
- `"likelihood"` is the original WordPiece scoring (pair count over the product of the part counts).

**CLI** (`DataTokenizerCommand.cs`):
- `--model unigram|wordpiece|bpe` picks the built-in trainer; `--trainer NAME` stays for plug-ins, and giving both is
  refused.
- New options: `--character-coverage`, `--byte-fallback`, `--max-piece`, `--seed-size`, `--shrinking`, `--em-steps`,
  `--split-digits`, `--lowercase`, `--strip-accents`, `--scoring`, `--arabic tatweel,diacritics`.
- The summary loses "(byte-pair encoding)", in English and Arabic.
- Progress shows each stage, how far it is, the speed and the time left (`ProgressLine`).

**Tests** (`tokenizer training:`, `cli pretrain:`):
- each trained tokenizer loads back and round-trips the corpus;
- `wordpiece` with `"frequency"` learns Hugging Face's vocabulary exactly;
- Unigram's vocabulary overlaps SentencePiece's (trained on the same corpus with the same options) above a threshold
  measured and recorded here;
- one worker equals many;
- the Arabic options strip tatweel and diacritics and Hugging Face reads the file;
- the CLI refuses `--model` with `--trainer` and bad values, and the `cli polish`, `cli health` and `cli arabic` checks
  pass.

**T4 as built (2026-10-19)** (agent B, on T1 merged with T2 and T3's GGUF work). As planned, with these differences
and measurements:

- **Contract** (`TokenizerTrainingOptions`):
  - The plan's fields, plus `UnknownToken` (null: "<unk>" for Unigram, "[UNK]" for WordPiece; after the special tokens
    when they do not hold it) and `Validate(trainer)`, which refuses a bad value naming the setting before a file is
    read. The trainers and the CLI call it; the CLI swaps the setting names for its option names.
  - `StripAccents` is `bool?`: null follows `Lowercase`, as BERT's normalizer does (the plan said `false`).
  - Which trainer reads which field: `CharacterCoverage`, `ByteFallback`, `MaxPieceLength`, `SeedSize`, `ShrinkingFactor`
    and `EmIterations` are Unigram's; `SplitDigits` and `Arabic` are both trainers'; `Lowercase`, `StripAccents` and
    `Scoring` are WordPiece's. `Arabic` on the bpe trainer is refused (it writes no normalizer).
  - Progress stages: "words", "seeds" (suffix-tree nodes), "em" (EM steps, the total estimated from the seeds, the
    vocabulary and the shrinking factor) and "prune" (rounds); "merges" for BPE and WordPiece.
  - **A new contract, `ITokenizerCompanionFiles`** (`Idrak.Models.Abstractions`): a tokenizer that also writes files
    besides tokenizer.json, each read back by its reader to the same tokenizer. The trained Unigram tokenizer writes
    tokenizer.model; the CLI writes every companion beside `-o`.
  - "unigram" and "wordpiece" are registered library defaults beside "bpe".
- **Unigram trainer** (`UnigramTrainer.cs`): sentencepiece 0.2.2's procedure, ported from its source distribution
  (PyPI's sdist; huggingface.co was not needed):
  - The text: lines normalized as its identity normalizer does (the Arabic removals; leading, trailing and repeated
    spaces dropped; a ▁ before the text; spaces as ▁). Special tokens written in a line become tabs, as SentencePiece
    cuts its meta pieces. Required characters come by coverage (compared in `float`), and the rest become ▅.
  - Seeds: SA-IS and esaxx's suffix-tree nodes (`SuffixArray.cs`), over the code points ranked to 1..k, so k is the
    alphabet and not 0x110000. Candidates go through its BoundedPriorityQueue, with its pruning at eight times the
    seed size, keyed by place in the text, so no substring becomes a string until it is kept.
  - Which pieces are valid: its `IsValidSentencePiece`, with sentencepiece's own Unicode script table
    (`UnicodeScripts.cs`, 1,287 ranges, generated by `tools/tokenizers/make_scripts.py` from `unicode_script_map.h`).
  - EM: its `float` lattice arithmetic, the digamma M step, NBest(2) pruning by loss, and its finalization. Its quirks
    are kept, such as the trainer's unknown node counting for id 0.
  - The E step and the pruning's Viterbi counts run in chunks of 1,024 words, whose sums are added in chunk order.
  - The tokenizer is what the sentencepiece reader makes of the ModelProto written by `SentencePieceWriter`. The writer
    gains the TrainerSpec fields 4, 10, 14, 15, 17, 20, 25 and 30. So tokenizer.json is in sentencepiece mode, and
    Hugging Face reads it too.
  - Special tokens: the given ones first, in order, as CONTROL pieces ("<s>", "</s>" and "<pad>" also as the bos, eos
    and pad ids), then the unknown token, then the byte pieces.
- **Not settable yet** (SentencePiece's defaults are kept, as fixed choices): `split_by_whitespace`,
  `split_by_unicode_script`, `split_by_number`, `treat_whitespace_as_suffix`, `allow_whitespace_only_pieces`, and the
  identity normalization (there is no NFKC rule; a `nmt_nfkc` charsmap is not built). No line is skipped for length:
  SentencePiece's `max_sentence_length` skips lines over 4,192 bytes. The algorithm's constants are also fixed: the
  desired size is 1.1 × the vocabulary, the expected-count floor is 0.5, and the unknown penalty is 10. These are
  candidates for settings when someone needs them.
- **Arabic:**
  - Unigram: a precompiled charsmap that removes the marks (`PrecompiledCharsMap.Build`, `DoubleArrayTrie.Charsmap`),
    so tokenizer.model holds the removals and sentencepiece reads them. tokenizer.json gets the same map as Precompiled
    in sentencepiece mode. Hugging Face's grapheme mode differs only where a removed tatweel carries a kept mark.
  - WordPiece: `Replace` normalizers before BertNormalizer.
- **Measured, Unigram against sentencepiece (same text, same options; `tokenizer training: unigram learns
  SentencePiece's vocabulary`):**
  - The three fixture configurations: vocabulary 1000; 1000 with byte fallback and split digits; 800 with the Arabic
    removals, longest piece 8, shrinking 0.8 and 3 EM steps.
  - On all three, the learned pieces are sentencepiece's set exactly: 100% overlap; 445, 520 and 256 of them at the same
    id; scores within 7.1e-5. **The floor is set at 98%** (`UnigramOverlapFloor`), to leave room for the platform's
    `Math.Log` and `Math.Exp`.
  - On 17 MB (168,000 lines of Arabic and English from `idrak data text`, vocabulary 8000): 99.92% overlap.
    sentencepiece's own one-thread and four-thread runs agree on 99.89%. Over 20,000 lines the two models encode to
    314,575 and 314,579 tokens.
  - One worker and four write byte-identical tokenizer.model files.
- **Speed and memory** (Release CLI, a 4-thread Linux container, the 17 MB text, vocabulary 8000):
  - Time: 20.1 s on one worker against sentencepiece's 23.5 s on one thread; 14.4 s against 15.3 s on four.
  - Memory: the live heap fits in 320 MB (out of memory at a 256 MB `DOTNET_GCHeapHardLimit`), against sentencepiece's
    390 MB RSS. The process RSS is 609 MB with the default GC: garbage it keeps while memory is plentiful, which a heap
    limit bounds.
- **Cross-check through the CLI** (`idrak tokenize` on each corpus line, against the references reading our files):
  - sentencepiece reads the trained tokenizer.model and gives our ids on every line, except the control pieces written
    in text (lines 55 and 56, T1's known difference).
  - tokenizers reads tokenizer.json and gives our ids on every line, except the float and double ties T1 found (the
    tatweel run on line 6, the 300 a's on line 47).
- **WordPiece trainer** (`WordPieceTrainer.cs`):
  - "frequency" is `BpeTrainer.Merge` (Hugging Face's do_train, its bookkeeping included) given a "##" prefix: a
    character after a word's first is its prefixed token, and a merge drops the right part's prefix. The ## characters
    are numbered after the alphabet in code-point order, the words are taken in code-point order, and ties pop as
    .NET's PriorityQueue pops them.
  - The fixture script ports this to Python, with a port of the 4-ary heap. The C# vocabulary equals it exactly (1,893
    tokens).
  - Hugging Face's own trainer is not repeatable: four runs agreed with each other on 96.5-97.9%, and with the fixed
    reference on 96.5-98.1%. The committed run overlaps 98.10%; **the floor is 95%**.
  - "likelihood": count(pair) / (count(left) × count(right)), ties to the smaller pair. Every pair of a merged part is
    queued again. One worker learns the same as four.
  - The tokenizer is laid out as BertConverter writes it. TemplateProcessing comes only when the special tokens hold
    [CLS] and [SEP]. `SplitDigits` adds `Digits` (individual) after BertPreTokenizer.
  - Hugging Face reads the file: with `--keep-accents`, its ids equal ours on 98 of 98 non-blank corpus lines,
    with the Arabic removals and split digits.
- **Found, outside T4: NFD does nothing in the CLI.** `src/Idrak.Cli` builds with `InvariantGlobalization=true`, and
  the library's NFD and NFKC (the span `UnicodeNormalizer`, through `TryNormalize`) need ICU. So in the CLI process,
  BERT's `strip_accents` does nothing:
  - A WordPiece tokenizer trained there with `--lowercase` keeps its accents, while Hugging Face strips them from the
    same file. 35 of 98 lines differ, and Hangul syllables decompose into [UNK] there.
  - The test process has ICU, so every test passes.
  - Rule 70: the fix is NFD, NFC and NFKC in C# with Unicode tables, in the library.
- **CLI** (`DataTokenizerCommand.cs`):
  - `--model bpe|unigram|wordpiece`; giving it with `--trainer` is refused.
  - New options: `--unk`, `--character-coverage`, `--byte-fallback`, `--max-piece`, `--seed-size`, `--shrinking`,
    `--em-steps`, `--split-digits`, `--arabic`, `--scoring`, `--lowercase`, `--strip-accents` and `--keep-accents`.
  - Default special tokens depend on the model: BERT's five for wordpiece.
  - The summary is "Learn a tokenizer from text files, ...", in English and Arabic. The result line names the model
    and the companion files.
  - Progress: one line per stage (`ProgressLine.NextStage`), each with its own rate and time left. The EM steps carry
    the pruning rounds in their label.
  - `IDRAK_TOKENIZER_FORMAT`, `_CACHE`, `_MAX_WORD_CHARS`, `_ALPHA`, `_NBEST` and `_SEED` joined the tool's
    environment table. They were missing, and the health test scans for every IDRAK_ name.
- **Tests:** seven new in `tokenizer training:`, and `cli pretrain:` extended. Fixtures are in
  `tests/Idrak.Tests/data/tokenizer-training/` (unigram.json 81 KB, wordpiece.json 30 KB, a hand-written README), from
  `tools/tokenizers/make_training_fixtures.py` (pinned requirements).

## Phase T5 (optional): BPE onto the shared pipeline

**On hold (the owner, 2026-10-19):** not started, measured or scheduled until the owner settles on it. No agent takes it as
ready work.

BPE keeps its own normalizer and pre-tokenizer fast paths until the span pipeline is measured against them: the same ids
on every fixture (rule 71), and equal or more tokens a second, with no more bytes allocated, on `--bench-text` (cold and
warm apart, rule 76). If both hold, BPE runs on the pipeline and its duplicate paths are removed (rule 75). If not, the
numbers go in the Runs table and BPE stays as it is.

**Measured, not taken (2026-10-19, the owner: "measure only").** A scratch program (not committed) ran BPE on the
span pipeline exactly as its detailed encode does, without offsets: `TokenizerPipeline`, then the same cached merge
(`AlphabetCache`, `Bpe`), added tokens first. Against BPE's own `EncodeRange` on all 26 BPE `tokenizer.json` files in
the test data, 967 corpus texts (60,781 characters), one thread, Release, an idle 4-thread container:
- **Ids:** equal on every text each vocabulary encodes (refused texts skipped alike on both paths).
- **Bytes:** 0 a text once warm on both paths.
- **Speed:** the span path is slower on every file: warm 0.19 to 0.85 of BPE's own speed, about 0.5 typically;
  cold 0.6 to 1.3.

  | tokenizer | own, warm | span, warm | ratio |
  |---|---|---|---|
  | tokenizers/bpe (sentencepiece-style, Metaspace) | 93.2M | 50.0M | 0.54 |
  | tokenizers/byte-level-roberta | 19.1M | 9.5M | 0.50 |
  | pretrain/tokenizer | 21.1M | 11.0M | 0.52 |
  | pretrain/metaspace | 10.3M | 3.9M | 0.37 |
  | gguf/tiny-llama-hf | 22.3M | 17.1M | 0.77 |
  | audio/tiny-whisper | 18.2M | 15.6M | 0.85 |
  | audio/ctc/model-conv | 26.9M | 5.0M | 0.19 |
  | vlm-qwen/tiny-qwen2.5-vl | 38.9M | 17.1M | 0.44 |
  | vlm/tiny-gemma3 | 386.3M | 157.8M | 0.41 |

- **Outcome:** the condition (equal or more tokens a second) fails, so BPE keeps its own paths and their duplicates.
  The gap is the span pipeline's per-character origin tracking, which an ids-only encode does not need. A span mode
  without origins would be the candidate if T5 is ever reopened.

---

## Finished after the handover (2026-10-19)

The handover's "What is left" (items 1-3) was done on the owner's "finish plan 24", one piece at a time, each merged
into `abstraction` after a build and its targeted CPU filters; then two speed rounds on what the new bench rows showed.
The merged tree (7b8c85f) passed 147 of 147 targeted CPU tests before the last push.

**1. Unicode normalization owned by the library** (bebfc2b, 63783ee; rule 70).
- **The bug, confirmed:** .NET's `string.Normalize` goes through ICU on Linux and macOS and through NLS on Windows. Under
  `InvariantGlobalization` (the idrak tool) it returns the text unchanged, so BERT's `strip_accents` kept accents, NFKC
  did nothing and Hangul became `[UNK]`. .NET 10's own character data (`CharUnicodeInfo`) is Unicode 16.0; this
  container's ICU 74 is Unicode 15.1.
- **The tables:** `tools/unicode/make_normalization_tables.py` reads the UCD 18.0.0 files (`UnicodeData.txt`,
  `DerivedNormalizationProps.txt`, `NormalizationTest.txt`; unicode.org is blocked here, raw.githubusercontent.com
  serves the unicodetools repository's `unicodetools/data/ucd/18.0.0/`), the same version as Imaging's text tables.
  It writes `src/Idrak.Abstraction/Text/UnicodeNormalization.Data.cs` (a two-stage table, 256 code points a page:
  110 pages, 3,948 records of combining class, quick-check and segment flags and full decompositions; 961
  composition pairs; longest decompositions 4 canonical, 18 compatibility) and the conformance file
  `tests/Idrak.Tests/data/unicode/normalization-test.txt.gz` (comments removed, 168 KB). Reruns write the same bytes.
- **`Idrak.Abstraction.Text.UnicodeNormalization`** (public): `IsNormalized`, `Normalize` (the same instance when
  normalized), `TryNormalize`, `GetNormalizedLength`, `Decompose`, `Compose`, `CombiningClass`,
  `MaxDecompositionLength`, `UnicodeVersion`. A quick check (a vectorized skip of the plain range, then one bit per
  basic-plane code unit, 8 KB a form, built once) finds the first segment that may change; only such segments are
  decomposed, reordered and composed, the rest copied. Nothing is allocated once warm; a lone surrogate is refused
  as .NET refuses it.
- **Every platform call in `src/` replaced:** the span normalizer (`SpanComponents.UnicodeNormalizer`: its alignments
  unchanged, the per-block stability table and its platform calls gone), BPE's string normalizers, the OpenType
  shaper, `AnswerTexts`, the tuning metrics.
- **Tests** (`tokenizers: unicode normalization:`, 6): every line of NormalizationTest.txt 18.0 in the four forms
  (20,171 lines, every other code point left as it is, the four APIs agreeing); equal to the platform on every code
  point, the corpora and every composition pair, except characters newer than the platform's Unicode (here NFC 40,
  NFD 20, NFKC 145, NFKD 125, counted and printed; on Windows the counts differ); no allocation; no platform call left
  in `src/` (a scan); the idrak tool's own process (its runtimeconfig checked for invariant globalization) strips
  accents and decomposes Hangul with this process's ids; speed printed. The pipeline allocation test now expects 0
  bytes on text NFKC changes (it allowed up to 2 KB, 624 B measured).
- **Speed** (61,743 characters of the corpora, median of 61 runs, fully optimized JIT, µs; the platform is ICU 74):

  | form | `Normalize`, ours | the platform's | `IsNormalized`, ours | the platform's |
  |---|---|---|---|---|
  | NFC | 190-195 | 236-387 | 5-8 | 39-83 |
  | NFD | 215-225 | 269-279 | 0.4-0.6 | 37 |
  | NFKC | 200-223 | 238-326 | 3.6-5 | 39 |
  | NFKD | 219-220 | 281-295 | 0.4-0.5 | 37-38 |

**2. The T3 callers** (7232737, 63783ee).
- **`idrak tokenize`** gains `--offsets` (UTF-16 start and end and the covered text), `--pair TEXT` (type and sequence
  ids), `--special` (the post-processor's tokens), `--max-length N` and `--truncation
  longest_first|only_first|only_second`; without them the output is byte for byte as before. Nothing is padded;
  without `--max-length` nothing is cut. Bad values are refused naming the option (exit 2). The summary is shorter
  (the help line limit), in English and Arabic.
- **`TokenizerEncodeOptions.TruncationDirection`** ("right", "left"; null: the tokenizer.json's): the direction came
  only from the file, so a file truncating from the left gave `TextEncoder` other ids.
- **`TokenizerEncodeOptions.Offsets`** (true by default): false sends a single text through the tokenizer's plain
  encode (its piece cache), with the same ids, type ids, masks, truncation, special tokens and padding, offsets
  (0, 0) and no token strings; a pair keeps the detailed path (tokenizers' pair truncation depends on its pre-tokens).
  Found by measurement: the detailed batch made `TextEncoder` slower than before and 13-17 times the memory.
- **`TextEncoder`** (Idrak.Nlp) encodes each batch with one `EncodeBatch` (ids only, right truncation to its
  `MaxLength`, no special tokens, not padded; it writes the pad ids and weights itself). A plain `ITokenizer` keeps
  the per-text path. The vectors are bit for bit those of the plain path (bpe, wordpiece, unigram fixtures; 1, 2, 3
  and all workers; empty, long, Arabic and emoji texts; a left-truncating, padding tokenizer.json).
- **A bug fixed on the way:** an app's tokenizer overriding a built-in (`TokenizerModels`' guard) encoded a batch one
  text at a time and never padded to the longest; each side's own `EncodeBatch` now runs, compared text by text.
- **Tests:** `retrieval: text encoder with a pretrained tokenizer …`, `cli run: tokenize --offsets, --pair,
  --special, …` (against `expected-hf.json` and `encodings.json`), the guarded batch in `tokenizers:`, and ids only
  against the detailed encode on every fixture (special tokens, truncation from either end, padding).

**3. The T4 leftovers** (c4fe91c, 67b8f7d).
- **The Unigram trainer's fixed SentencePiece choices are settings** (`TokenizerTrainingOptions`, checked by
  `Validate`, written into tokenizer.model's TrainerSpec fields 18, 21-24, 26 and the NormalizerSpec):

  | setting | default | `idrak data tokenizer` option |
  |---|---|---|
  | `SplitByWhitespace` | true | `--no-whitespace-split` |
  | `SplitByUnicodeScript` | true | `--no-script-split` |
  | `SplitByNumber` | true | `--no-number-split` |
  | `TreatWhitespaceAsSuffix` | false | `--whitespace-suffix` |
  | `AllowWhitespaceOnlyPieces` | false | `--whitespace-pieces` |
  | `MaxSentenceLength` | 0, no limit | `--max-sentence` |
  | `Normalization` | `identity` (also `nfkc`, `nmt_nfkc`) | `--normalization` |
  | `DesiredSizeFactor` | 1.1 | `--desired-size` |
  | `ExpectedCountFloor` | 0.5 | `--expected-floor` |
  | `UnknownPenalty` | 10 | `--unknown-penalty` |

  - `MaxSentenceLength` defaults to no limit, not sentencepiece's 4,192 bytes, so no corpus loses text it kept before
    (a paragraph a line of Arabic often passes 4,192 bytes); no limit is written as sentencepiece's maximum.
  - `nfkc` and `nmt_nfkc` are built in C# (`SentencePieceRules.cs`: sentencepiece's `BuildNFKCMap`, `MergeNmtMap`,
    `RemoveRedundantMap` on `UnicodeNormalization`), once a process; the Arabic removals fold into the same map.
    Against sentencepiece's built-in maps, on all 1,112,063 code points, 1,839 combining sequences and every corpus
    line: equal but 68 code points Unicode 18 maps and sentencepiece's older data does not (listed in
    `tokenizer-training/normalization.json`).
  - Each of the eight switch configurations learns sentencepiece's pieces, 100%; `--whitespace-pieces` all 143 at the
    same ids. sentencepiece reads our nmt_nfkc and whitespace-suffix models to our ids on 100 of 102 lines (the two
    control-piece lines); one nmt_nfkc line is an exact float tie ("000"), accepted only as a proven tie.
  - `DoubleArrayTrie`'s builder: 3.5 s → 0.4 s for the 225,000-key NFKC map (it checked every position of nearly
    full blocks; the bases are identical).
  - **Not done:** the `_cf` rules (no case-folding table); `add_dummy_prefix` and `remove_extra_whitespaces` stay fixed.
- **`--bench-text`** takes sections (`text`, `tokenizers`, `trainers`) and the variables `IDRAK_BENCH_TEXT_CHARACTERS`
  (200,000), `_SECONDS` (1), `_LOADS` (5), `_WORKERS` (0), `_TOKENIZERS`, `_REFERENCE` (`make_fixtures.py --bench`'s
  JSON), `_TRAIN_FILES`, `_TRAIN_MB` (16), `_TRAIN_VOCAB` (3,000), `_TRAINERS`. Rows: every fixture tokenizer, ids,
  detailed, batch detailed and ids only on 1 and all workers, cold and warm, bytes a text, load time and memory; the
  three trainers on 1 and all workers.

**4. Speed round one** (846b028), from the new rows; results proven unchanged by a dump of every encoding of every
fixture (every path and option, batches on 1 and 4 workers, cache on and off) and of BPE and WordPiece vocabularies
trained on 1 and 4 workers, before and after (identical but the message text of the `AggregateException`s of
4-worker batches over refused texts, a race in how many failures are aggregated).

| row | before | after |
|---|---|---|
| BPE `EncodeRange` into a reused list (bpe model / pretrain json) | 119 / 243 B a text | 0 B |
| ids-only batch, bpe json, 1 / 4 workers | 22.2M / 65.9M tok/s, 3,975 B | 37.4M / 99.8M, 1,759 B |
| detailed BPE, bpe json, 1 thread | 4.5M, 10,357 B | 14.4M, 2,210 B |
| detailed BPE batch, 1 → 4 workers | 2.6M → 4.3M | 14.2M → 48.1M |
| BPE trainer, 4 workers (16 MB) | 0.65-0.74 s, 506-575 MB | 0.14-0.18 s, 13.3 MB |
| WordPiece trainer, 4 workers | 0.56-0.64 s, 206-210 MB | 0.29-0.32 s, 5.3 MB |

The trainers' waste was regex contention: one `Regex` shared by threads builds a matcher per match on every other
thread.

**5. Speed round two** (9756572), proven unchanged the same way.
- **Regex contention in encoding:** every tokenizer pattern is a `TokenizerRegex` (compiled once, a per-thread copy
  sharing the compiled matcher, about 6 µs a thread). A 4-worker batch now allocates within 1.1 B a text of one
  worker (a test, tolerance 32 B); wordpiece-uncased ids-only on 4 workers 9.4M → 44.1M tok/s, bpe json detailed
  44.6M → 74.4M.
- **WordPiece:** a one-pass BERT normalizer and pre-tokenizer (`BertComponents.OnePass`, its character classes built
  from the steps' own rules; context-dependent text falls back to the steps), and the shared `PieceCache` (no
  `ConcurrentDictionary.Count` per miss; it keeps token ranges for the detailed encode). Uncased ids 3.96M → 17.0M,
  cased 6.77M → 21.3M, detailed 2.94M → 9.9M; training on 16 MB, 1 / 4 workers, 1.37 / 0.36 s → 0.67 / 0.27 s.
- **Detailed BPE:** caches serving the detailed encode keep each token's range beside its ids; in-place one-character
  Replace and Lowercase. bpe json detailed 13.8M → 19.5M.
- **One cache budget:** a byte-level tokenizer's two caches together hold at most `PieceCacheLimit` pieces (before,
  up to twice that).

## What is left (2026-10-19, after 7b8c85f)

1. **The owner's runs** ("The owner's runs"): the targeted Windows CPU list, CUDA `pretrained:|gguf:|tokenizers:
   gguf|retrieval: text encoder`, the CLI on real models, `make_fixtures.py --real` and `tokenizers real:`, and
   `--bench-text tokenizers trainers`. Their results go in the Runs table.
2. **T5** (BPE onto the shared span pipeline): measured 2026-10-19 (its section): the same ids, about half the speed;
   not taken. Reopened only by the owner.
3. **Measured, not taken** (each a candidate for the owner to schedule; none is started without the owner):
   - detailed BPE is still 1.8 times the ids-only encode (a per-token origin lookup; `TokenizerEncodings.Resolve`
     reads the tokenizer.json's settings on every encode);
   - the byte-level span pre-tokenizer costs 42 of 124 ns a character on the detailed byte-level path;
   - Unigram's detailed encode realigns every cache hit (WordPiece's cached ranges would apply);
   - each tokenizer compiles its patterns on load (about 6 ms for the long ones);
   - an nfkc or nmt_nfkc tokenizer.model is about 2 MB against sentencepiece's 240 KB (`DoubleArrayTrie` does not
     merge shared suffixes);
   - the Unigram trainer gains little from workers on the repeated fixture corpus (7.1 → 6.1 s on 16 MB) and peaks at
     about 270 MB of heap;
   - an ids-only encoding still holds the contract's six arrays (about 1.1-1.8 KB a text in a batch);
   - truncation `stride` (overflowing windows) is not implemented.

## Speed and memory

- `--bench-text` (`tests/Idrak.Tests/BenchText.cs`) gains rows for Unigram, WordPiece and SentencePiece-BPE on the
  corpus: tokens a second (cold and warm apart), bytes allocated per encode once warm (the target is 0 into a reused
  list), and load time and memory for the largest real vocabulary.
- The reference speed (`tokenizers`, one thread and all threads) is measured by `make_fixtures.py --bench` on the same
  corpus and machine. Both go in the Runs table.
- Load: vocabularies are read from bytes into flat arrays; tries are built once a load. The load time joins the
  existing tokenizer timing in `PretrainedModel` (`tokenizing`).

## Order and parallelism

At most two agents at a time (CLAUDE.md), each in its own worktree, building only its projects:

| Wave | Agent A | Agent B |
|---|---|---|
| 1 | L0 | T0, the part that needs nothing else: the fixture script and trained fixtures, `ProtoReader`, `DoubleArrayTrie`, `ModelProto` parsing |
| 2 | L1 | T0, the rest: the `sentencepiece` reader registered (after L0), the Precompiled lookup; the GGUF reader's `byte[]` / `float[]` arrays |
| 3 | T1 (Unigram; Precompiled's alignments on L1) | T2 (WordPiece, BERT components, post-processors, `vocab.txt`) |
| 4 | T3a: encodings, offsets, pairs, batch | T3b: detection, GGUF `llama` / `bert` / `t5`, GGUF fixtures |
| 5 | T4: the Unigram trainer, `SentencePieceWriter` | T4: the WordPiece trainer, then the CLI |
| 6 | T5: on hold until the owner decides | none |

The table is a dependency order, not a barrier (the owner, 2026-10-19): the moment an agent finishes, its slot takes
the next work whose inputs are merged, without waiting for the other slot. Work that needs only wave 1 and can start
early: the Unigram and WordPiece model cores (piece matching over `DoubleArrayTrie`, before L1's pipeline is merged),
the GGUF fixture tokenizers (`tools/gguf/make_fixtures.py`), the trainers' suffix array. The critical path is L0, L1,
T1 or T2, T3, T4. The owner's `make_fixtures.py --real` can run as soon as wave 1 is
merged. Files both agents of a wave touch (`api/*.txt`, the inventory, `TokenizerComponents.cs`, `Tests.All`,
`EnvironmentVariables.cs`, the decoder count in `ModelPluginTests.cs`) are merged one agent at a time; the generated
dumps are regenerated after a conflicting merge, never merged by hand, and that merge gets one build and the
`public API` and `inventory` filters.

Each phase follows the owner's fix workflow: fix everything, update the tests it touches, build the changed projects
once, run its filter once with `--no-build` (CPU here), merge and push to `abstraction`, and hand over the owner's
commands.

## The owner's runs

Tokenizers run on the CPU, so a phase's GPU list is the end-to-end load only. From `D:\Projects\Idrak`:

```powershell
# once, after a phase is merged
dotnet build -c Release tests\Idrak.Tests

# T0 and later: the real models' references (downloads into the cache; writes tests\Idrak.Tests\data\tokenizers\real)
python -m pip install -r tools\tokenizers\requirements.txt
python -I tools\tokenizers\make_fixtures.py --real
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "tokenizers real:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER

# T3: GGUF tokenizers on CUDA, and a T5 and a BERT folder through the CLI
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "gguf:|tokenizers: gguf"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
dotnet run -c Release --project src\Idrak.Cli -- tokenize google-t5/t5-small "Hello, مرحبا 😀" --offsets
dotnet run -c Release --project src\Idrak.Cli -- tokenize google-bert/bert-base-uncased "Hello, World!" --pair "Second" --special
# a GGUF whose tokenizer.ggml.model is llama (Llama-2-style), end to end
dotnet run -c Release --project src\Idrak.Cli -- chat PATH\TO\MODEL.gguf

# T4: the trainers (CPU only; no GPU list), and one tokenizer of each kind from the CLI
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "tokenizer training|cli pretrain|tokenizers: unigram|public API|abstraction inventory"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
dotnet run -c Release --project src\Idrak.Cli -- data tokenizer corpus.txt --model unigram --vocab 8000 --byte-fallback -o out\unigram\tokenizer.json
dotnet run -c Release --project src\Idrak.Cli -- data tokenizer corpus.txt --model wordpiece --vocab 30000 --arabic tatweel,diacritics -o out\wordpiece\tokenizer.json

# speed (T1 and later)
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-text
python -I tools\tokenizers\make_fixtures.py --bench
```

After the handover's items (7b8c85f), everything changed since 48d058d in one list:

```powershell
git checkout abstraction; git pull
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "tokenizers:|tokenizer training|cli pretrain|pretrained:|gguf:|plugins:|override loop:|abstraction inventory|public API|cli health: every environment variable|cli arabic:|cli polish|cli run:|retrieval: text encoder|shaping:|answer texts:|vision tuning contracts: cer and wer"
dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "pretrained:|gguf:|tokenizers: gguf|retrieval: text encoder"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
dotnet run -c Release --project src\Idrak.Cli -- tokenize google-t5/t5-small "Hello, مرحبا 😀" --offsets
dotnet run -c Release --project src\Idrak.Cli -- tokenize google-bert/bert-base-uncased "Café Résumé" --pair "Second" --special
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-text tokenizers trainers
```

On Windows, `tokenizers: unicode normalization: equal to the platform's normalization` compares with Windows' NLS:
differences on characters newer than its Unicode are counted and printed, any other fails.

## Risks

- **Two references disagree.** `sentencepiece` and `tokenizers` differ on a few inputs for the same model (whitespace
  runs, some graphemes). Each path is held to its own reference, and the differences are listed, not hidden.
- **Grapheme rules.** Precompiled works by grapheme cluster; .NET's text elements and Rust's `unicode-segmentation` may
  disagree on the newest emoji sequences. The corpus has them; a difference is fixed in our segmentation or listed.
- **Unicode versions.** NFKC data in .NET and in the references can differ for new characters; the fixture manifest
  records the Unicode versions.
- **Unigram training equal to SentencePiece** is not a goal (its seeding and pruning have tie orders of their own): the
  check is overlap above a measured threshold, as the spec says.
- **The API break** in L0 reaches the samples and plug-ins that name `BpeTokenizer` in a type. `BpeTokenizer` keeps its
  members, so a cast or the interface fixes each.

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
| 2026-10-19 | T1 | 4-thread Linux container, CPU | `tokenizers:`, then `plugins:`, `pretrained:`, `gguf:`, `tokenizer training:`, `override loop:`, `public API`, `inventory` | 29 and 49 passed; first run 5 failures (empty-text Metaspace, suite round trip on `<unk>`, sampling variety on `unigram-bytes`, API, inventory), fixed and rerun |
| 2026-10-19 | T1 | same | `make_fixtures.py --bench` against `tokenizers: unigram: speed` | the table in T1 as built |
| 2026-10-19 | T3 encodings | CPU (cloud, 4 threads) | `tokenizers:`, `abstraction inventory`, `public API` | 42 passed, 0 failed |
| 2026-10-19 | T1 merged with T3 | CPU (cloud, 4 threads) | `tokenizers:`, `tokenizer training`, `pretrained:`, `gguf:`, `plugins:`, `abstraction inventory`, `public API` | 84 passed, 0 failed |
| 2026-10-19 | T4 merged with T1 and T3 | CPU (cloud, 4 threads) | `tokenizers:`, `tokenizer training`, `cli pretrain`, `pretrained:`, `gguf:`, `plugins:`, `abstraction inventory`, `public API`, `cli health: every environment variable`, `cli arabic:` | 103 passed, 0 failed |
| 2026-10-19 | T4 | 4-thread Linux container, CPU | `tokenizer training`, `cli pretrain`, `tokenizers: unigram`, `public API`, `abstraction inventory`, and the CLI's environment, help and Arabic-catalogue checks | 30 passed, 3 failed (three tests gave a vocabulary smaller than the corpus' 655-character alphabet); after the fix, `tokenizer training` and `cli pretrain: data tokenizer`: 13 passed, 1 failed (a decode expectation that missed T1's added-token segments); that test: 1 passed |
| 2026-10-19 | T4 | same | the unigram trainer against sentencepiece 0.2.2 on 17 MB, vocabulary 8000, Release CLI | 20.1 s against 23.5 s (one thread), 14.4 s against 15.3 s (four); live heap within 320 MB against 390 MB RSS; 99.92% of its pieces |
| 2026-10-19 | T4 | same | the trained files read by the references (`idrak tokenize` per corpus line) | unigram: sentencepiece and tokenizers give our ids but on T1's known lines; wordpiece: tokenizers equal on 98/98 with `--keep-accents`, 63/98 with accents stripped (the CLI's NFD under invariant globalization) |
| 2026-10-19 | 1 (normalization) | CPU (cloud, 4 threads) | `tokenizers:`, `tokenizer training`, `cli pretrain`, `pretrained:`, `gguf:`, `plugins:`, `abstraction inventory`, `public API`, `cli health: every environment variable`, `cli arabic:`, `shaping:`, `answer texts:`, `vision tuning contracts: cer and wer` | 113 passed, 3 failed (the platform comparison's tolerance missed Todhri, new in 16.0; a CLI check assumed a "caf" token; the API dump); after one rebuild the normalization, pipeline, wordpiece and encoding groups 20 of 20 |
| 2026-10-19 | 1 and 2 merged | same | the list above with `cli run:`, `cli polish`, `retrieval: text encoder`, `override loop:` | 140 passed, 0 failed; with ids only: 139 and 1 (an empty text's pad and special tokens), then `tokenizers: encoding` 5 of 5 |
| 2026-10-19 | 1 | same, a quiet scratch bench | `UnicodeNormalization` against ICU 74 | the table in "Finished after the handover", 1 |
| 2026-10-19 | 2 (agent) | same | `tokenizers:`, `pretrained:`, `cli run:`, `cli polish`, `cli arabic:`, `cli health`, `public API`, `inventory`, `retrieval: text encoder` | 79 passed, 2 failed (a help line over 120 characters, the API dump); 11 of 11 after |
| 2026-10-19 | 3 (agent) | same | `tokenizer training`, `cli pretrain`, `tokenizers:`, `public API`, `abstraction inventory`, `cli polish`, `cli arabic:`, `cli health` | 84 passed, 4 failed (inventory, a wrong nmt_nfkc check, a missing folder, a float tie); 70 and 1 of 1 after |
| 2026-10-19 | 3 (agent) | same | `--bench-text` (Release) and `make_fixtures.py --bench` | the tables in commit 67b8f7d |
| 2026-10-19 | speed 1 (agent) | same | `tokenizers:`, `tokenizer training`, `pretrained:`, `gguf:`, `cli pretrain`, `retrieval: text encoder`, `cli run: tokenize`, `public API`, `abstraction inventory` | 87 passed, 0 failed; after a trainer change `tokenizer training|cli pretrain` 17 of 17 |
| 2026-10-19 | speed 2 (agent) | same | the speed 1 list with `plugins:`, `override loop:` | 112 passed, 0 failed |
| 2026-10-19 | all merged (7b8c85f) | same | the owner's CPU list above | 147 passed, 0 failed |
| 2026-10-19 | T5 (measure only) | CPU (cloud, 4 threads, idle) | BPE on the span pipeline against its own `EncodeRange`, 26 BPE tokenizer.json files, 967 texts | same ids and 0 B a text on both; the span path 0.19-0.85 of the speed warm (about 0.5): not taken |

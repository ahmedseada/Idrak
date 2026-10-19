# Plan 24: Unigram (SentencePiece) and WordPiece tokenizers, built in

**Status (2026-10-19, `abstraction`):** planned 2026-10-19 from the owner's spec (plan 24, draft 1) against e53acb8.
L0, L1, T0 and T2 are built and CPU-checked; T1 is in progress. Closes plug-in gap 11
([plug-in.md](plug-in.md), "Only BPE tokenizer models").

| Phase | What | State |
|---|---|---|
| L0 | The tokenizer model registry, one tokenizer contract, one loader; callers off `BpeTokenizer` | Done; CPU pass (155 tests) |
| L1 | The span pipeline with alignments: built-in components on spans, the post-processor table | Done; CPU pass |
| T0 | Reference fixtures; the SentencePiece `tokenizer.model` reader; the Precompiled normalizer | Done; CPU pass |
| T1 | Unigram: encode, decode, byte fallback, sampling | In progress (agent B) |
| T2 | WordPiece: the BERT components, greedy longest match, `[CLS] … [SEP]` and pairs | Done; CPU pass |
| T3 | Encodings (offsets, type ids, masks, pairs, batch), detection across files, GGUF `llama` / `bert` / `t5` | Detection done (T0, T2); GGUF `llama` / `bert` in progress; encodings and `t5` after T1 |
| T4 | Trainers: Unigram (EM), WordPiece; `idrak data tokenizer --model` | Planned |
| T5 | BPE onto the shared span pipeline (only if measured equal or faster) | On hold: not started until the owner decides (2026-10-19) |

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

**Callers**: `TextEncoder` (Idrak.Nlp) pads and masks through `EncodeBatch` when its tokenizer is an
`IPretrainedTokenizer` (the same vectors, one tokenize pass); `idrak tokenize` gains `--offsets`, `--pair TEXT` and
`--special`.

**Tests** (`tokenizers: encoding`, `tokenizers: gguf`):
- offsets, type ids and masks equal the reference for every fixture (BPE included);
- truncation and padding equal the reference;
- batch encoding equals single encoding for 1 and many workers;
- each GGUF fixture gives the ids of the Hugging Face files it was made from;
- `gguf:` passes with `FormatVersion` 2.

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

## Phase T5 (optional): BPE onto the shared pipeline

**On hold (the owner, 2026-10-19):** not started, measured or scheduled until the owner settles on it. No agent takes it as
ready work.

BPE keeps its own normalizer and pre-tokenizer fast paths until the span pipeline is measured against them: the same ids
on every fixture (rule 71), and equal or more tokens a second, with no more bytes allocated, on `--bench-text` (cold and
warm apart, rule 76). If both hold, BPE runs on the pipeline and its duplicate paths are removed (rule 75). If not, the
numbers go in the Runs table and BPE stays as it is.

---

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

# speed (T1 and later)
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-text
python -I tools\tokenizers\make_fixtures.py --bench
```

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
| | | | | |

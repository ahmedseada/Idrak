# Idrak.Text.Plugins

The text model families: decoders, encoders and encoder-decoders, with their GGUF mappings, tool-call formats and RoPE
scalings. The library (`Idrak`, `Idrak.Text`) keeps only generic defaults and fallbacks; the families live here, one
namespace each (`Idrak.Text.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Text.

Nothing is registered: finding a checkpoint's family by its architecture name (`PretrainedModel.Load`,
`PretrainedEncoderModel.Load`, `PretrainedSeq2SeqModel.Load`, a GGUF file's architecture), and picking a tool-call format,
a fill-in-the-middle format, a RoPE scaling or a coding language by its name, are not on the new design yet, so loading a
model stops there.

## Families

### BERT (`Idrak.Text.Plugins.Bert`)

The `BertPlugin` family holds `BertModel`, `BertForMaskedLM` and `BertForSequenceClassification` (config.json's
`architectures`); `PretrainedEncoderModel.Load` reads a Hugging Face folder through it (safetensors
or a PyTorch checkpoint, with or without the `bert.` prefix, LayerNorm `weight`/`bias` or the older `gamma`/`beta`):
word, learned position and token-type embeddings, post-norm blocks (`TransformerEncoderLayer` with
`NormPlacement.After`), and the heads the checkpoint holds: `pooler` (dense + tanh of the first token),
`masked-lm` (the transform and its LayerNorm, the decoder reading the word embeddings in place unless
`tie_word_embeddings` is false) and `sequence-classification` (the pooler, then the classifier). Its tokenizer is the
folder's WordPiece (`tokenizer.json` or `vocab.txt`).

```csharp
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Bert;
using Idrak.Text.Retrieval;
using var bert = PretrainedEncoderModel.Load(folder);
using (Autograd.NoGrad())
{
    var logits = bert.Logits(EncoderHeads.MaskedLanguageModel, ids, lengths, tokenTypes);   // [batch, time, vocabulary]
}

// Sentence vectors: the mean of the real tokens' states ("mean") or the [CLS] state ("first"), length 1.
var sentences = new TextEncoder(bert.Encoder, bert.Tokenizer!, 128, bert.Tokenizer!.IdOf("[PAD]") ?? 0) { Pooler = "mean", SpecialTokens = true };
float[][] vectors = sentences.Encode(texts);
```

The heads read the first token (the family's pooler, `PretrainedEncoderModel.Pooler`); choose a sentence vector's
pooling on `TextEncoder`, not with `PretrainedEncoderOptions.Pooler`, which the classifier reads too. Relative position
embeddings (`position_embedding_type` other than `absolute`) and decoder checkpoints (`is_decoder`) are refused.

### T5 (`Idrak.Text.Plugins.T5`)

The `T5Plugin` family holds `T5ForConditionalGeneration` and `MT5ForConditionalGeneration`,
and `T5EncoderModel` and `MT5EncoderModel`. `PretrainedSeq2SeqModel.Load` reads a Hugging
Face folder through it: the shared token embedding (the encoder's input, the decoder's input and, tied, the head's table, one copy),
pre-norm blocks with RMS norms (`TransformerEncoderLayer` with RMS norms; `DecoderBlock` with cross-attention), attention
without score scaling, the library's `bucketed` relative position bias of each stack's first block shared by the
others, the ReLU feed-forward (T5 v1.0) or the gated GELU one (`feed_forward_proj` `gated-gelu`: v1.1, mT5), and the
head: the shared embedding with the decoder's output scaled by `d_model^-0.5` (tied, v1.0) or `lm_head.weight`
(`tie_word_embeddings` false). Decoding starts from `decoder_start_token_id` and ends at `eos_token_id`; the longest
target is `n_positions` (512 when not given). The tokenizer is the folder's Unigram (`tokenizer.json`, or
`spiece.model`, whose missing `</s>` the generator appends).

```csharp
using Idrak.Models;
using Idrak.Text.Plugins.T5;
using var t5 = PretrainedSeq2SeqModel.Load(folder);
var generator = t5.CreateGenerator();
var (text, reason, stats) = generator.Generate("translate English to German: The house is wonderful.",
    new GenerationOptions { NumBeams = 4, NumPredict = 64 });          // greedy: TopK = 1, RepeatPenalty = 1
```

In the inference engine, an encoder-decoder is the `seq2seq` kind (`engine.Seq2SeqModel(name, folder)`, reached as
`ITextModel`); `idrak run FOLDER "prompt"` answers with it (`--beams N` for beam search). Other `feed_forward_proj`
values, bucket settings that do not fit and checkpoints without a tensor the configuration names are refused naming them.

### Gemma (`Idrak.Text.Plugins.Gemma`)

The `GemmaPlugin` family holds `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM` and the text decoder of
`Gemma3ForConditionalGeneration` (`GemmaArchitectures`); `PretrainedModel.Load` reads
them through it on the library's decoder (`DecoderSpec`): gains stored as a difference from 1, embeddings scaled by √width and
tied, and for Gemma 2 and 3 the norms after attention and the feed-forward block too (`GemmaArchitectures.TensorName`),
scores scaled by `query_pre_attn_scalar`, windowed layers alternating as the configuration or `layer_types` says,
soft-capping (Gemma 2), q/k norms and the windowed layers' own rotary base (Gemma 3, transformers 5's
`rope_parameters` too). Gemma writes its configuration from a spec (`PretrainedModel.Create`); Gemma 2 and 3 are read
from checkpoints only. The text decoder of Gemma 3 with images reads `text_config` (with transformers' defaults for
absent keys), never soft-caps its logits, and fits each checkpoint's naming (`language_model.model.` or
`model.language_model.`); its vision part is the Gemma 3 vision plug-in of `Idrak.Vision.Plugins`, which builds on
this family.

It also holds llama.cpp's `gemma`, `gemma2` and `gemma3` (`GemmaGguf`), read as
`GemmaForCausalLM`, `Gemma2ForCausalLM` and `Gemma3ForCausalLM`: llama.cpp's converter stores every norm gain with
Gemma's 1 added, taken back off as the gains are read (`GgufArchitecture.StoredOffset`); Gemma 2 and 3 keep their
post-norms under llama.cpp's names (`post_attention_norm`, `ffn_norm` before the feed-forward block, `post_ffw_norm`;
`GgufArchitecture.TensorName`); the feed-forward block is GELU (tanh); the sliding window, its layer pattern,
soft-capping and the windowed layers' RoPE base come from the file when it has them, else as llama.cpp assumes; and the
query scalar, which the files do not store, is llama.cpp's (`GemmaGguf.ImpliedQueryScalar`; `GemmaGguf.Gemma2` and
`Gemma3` take another). The tokenizer is the file's SentencePiece one.

```csharp
using Idrak.Models;
using Idrak.Text.Plugins.Gemma;
using var model = PretrainedModel.Load(folder);
```

### Llama (`Idrak.Text.Plugins.Llama`)

The `LlamaPlugin` family holds `LlamaForCausalLM` (`LlamaArchitectures.Llama()`: the
library's common configuration keys and checkpoint names, `PretrainedFamilies.CommonArchitecture`), llama.cpp's `llama`
(each head's query and key rows put back in order, experts read as `MixtralForCausalLM`, and Llama
3's RoPE scaling recovered from `rope_freqs.weight`), the GGUF pre-tokenizer names `llama3`, `llama-bpe`, `llama-v3`,
`smaug-bpe`, `falcon3` and `pixtral`, the RoPE scaling `llama3` (`LlamaRopeScaling`), and
Llama 3's chat template for a GGUF file without one whose vocabulary holds `<|start_header_id|>`, `llama3` in
`GgufChatTemplates` (after the library's ChatML).

### Qwen (`Idrak.Text.Plugins.Qwen`)

The `QwenPlugin` family holds `Qwen2ForCausalLM` (query, key and value biases), `Qwen3ForCausalLM` (q/k norms),
`Qwen2MoeForCausalLM` and `Qwen3MoeForCausalLM` (experts, a gated shared expert; `QwenArchitectures.MoeTensorName`), llama.cpp's `qwen2`, `qwen3`, `qwen2moe` and `qwen3moe`, the GGUF
pre-tokenizer names `qwen2`, `qwen35`, `megrez` and `hunyuan`, and the XML tool-call format `qwen3-coder`. The vision-language Qwen models are `Idrak.Vision.Plugins`'.

### Mistral (`Idrak.Text.Plugins.Mistral`)

The `MistralPlugin` family holds `MistralForCausalLM` and `MixtralForCausalLM` (experts under `block_sparse_moe`,
`MistralArchitectures.MixtralTensorName`), the GGUF pre-tokenizer `tekken`, and the
tool-call format `mistral` (`[TOOL_CALLS]`). Mistral's GGUF files are llama.cpp's `llama`, so it builds on the Llama
plug-in.

### DeepSeek and harmony (`Idrak.Text.Plugins.DeepSeek`, `Idrak.Text.Plugins.Harmony`)

The `DeepSeekPlugin` family holds DeepSeek's special-token tool-call format `deepseek` and the GGUF pre-tokenizer
`deepseek-r1-qwen` (Qwen2's rule); the `HarmonyPlugin` family holds the harmony tool-call format `harmony`
(reasoning in the analysis channel, calls as messages to `functions.name`). The tool-call parsers build on the library's
`MarkedToolCallParser` and `ToolCallText`.

```csharp
using Idrak.Models;
using Idrak.Text.Plugins.Qwen;
using var model = PretrainedModel.Load(folder);              // or GgufModel.Prepare("model.gguf") first
```

### Fill-in-the-middle formats

Four families have a fill-in-the-middle format (`<Family>Plugin.FillInMiddle`), each on the library's
`FillInMiddleLayout` with the tokens of its own tokenizer; detecting a model's format from its tokenizer is not on the
new design yet:

| Format | Family | Layout | Repository files |
|---|---|---|---|
| `qwen` | Qwen2.5-Coder, Qwen3-Coder | `<\|fim_prefix\|>prefix<\|fim_suffix\|>suffix<\|fim_middle\|>` | `<\|repo_name\|>` and `<\|file_sep\|>path` before it |
| `deepseek` | DeepSeek-Coder | `<｜begin▁of▁sentence｜><｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>` | `#path` lines at the start of the prefix |
| `gemma` | CodeGemma | `<bos><\|fim_prefix\|>prefix<\|fim_suffix\|>suffix<\|fim_middle\|>` | `<\|file_separator\|>path` before it |
| `mistral` | Codestral | `<s>[SUFFIX]suffix[PREFIX]prefix` (suffix first, no middle token) | none |

CodeLlama (Llama's family) has none: Meta's infilling feeds `▁<PRE>`, `▁<SUF>` and `▁<MID>` by id with the
suffix encoded without SentencePiece's leading space, which a text prompt does not express; an app that has checked its
tokenizer makes its own `FillInMiddleLayout`.

```csharp
var tokenizer = (IPretrainedTokenizer)model.Tokenizer!;
var format = QwenPlugin.FillInMiddle;                          // the model's family's format, reached directly
var prompt = format.Create(new FillInMiddleContext(tokenizer, before, after));
await foreach (var chunk in generator.StreamAsync(prompt.Text, FillInMiddleFormats.Options(prompt, options), token)) { … }
```

## Coding languages

The library's coding agent and tools (`CodingAgent`, `CodingTools`, Idrak.Text) name no programming language: they
compose their defaults from the coding languages (`ICodingLanguage`: `Idrak.Text.Abstractions`); picking them by
name is not on the new design yet. This package has the library's own, each a plug-in in a namespace of its own:

| Namespace | Plug-in | Languages | What they add |
|---|---|---|---|
| `Idrak.Text.Plugins.CSharp` | `CSharpLanguagePlugin` ("csharp-language") | `csharp` (a `*.sln`, `*.slnx` or `*.csproj`) | `dotnet`; checked with `dotnet build`, `dotnet test`; `bin`, `obj`, `.vs` skipped; tasks without commands run `dotnet restore`, `dotnet build --no-restore`, `dotnet test --no-build` |
| `Idrak.Text.Plugins.TypeScript` | `TypeScriptLanguagePlugin` ("typescript-language") | `typescript` (`package.json`, `tsconfig.json`), `angular` (`angular.json`) | `npm`, `npx`, `node`, `tsc` (and `ng`); checked with `npx tsc --noEmit`, `npm test` (`npx ng build`, `npx ng test --watch=false`); `node_modules` linked into a task's workspace, `dist` and `.angular` skipped |
| `Idrak.Text.Plugins.Python` | `PythonLanguagePlugin` ("python-language") | `python` (`pyproject.toml`, `setup.py`, `requirements.txt`, …) | `__pycache__` and `.venv` skipped (no commands) |

With all three and every language chosen (`CodingToolOptions.Languages`, `IDRAK_CODING_LANGUAGES`: `all`, the
default), the agent's prompt, its programs and the tools' descriptions are what the library had before the languages
moved out; `detected` keeps those the workspace shows. A workspace no chosen language detects gets a notice
(`CodingTools.LanguageNotice`) naming the plug-in that has one.

### C# task families (`idrak data code`)

The `CSharpLanguagePlugin` plug-in also has the C# task families of plan 36 (Idrak.Data's `ITaskFamily`), both
declaring `csharp` (`CSharpTaskFamilies`):

| Family | Draft | Check |
|---|---|---|
| `method-body` | a method's body removed (it throws `NotImplementedException`), its signature and `///` documentation kept | the project's tests that reach the method (up to 8) |
| `test-subject` | the code a named test entry `("group: what it checks", d => …)` reaches removed | that test, run by its name through the runner's filter variable (`IDRAK_FILTER`, with `IDRAK_DEVICES=cpu`) |

A test reaches a method it calls by name, directly or through a helper of its test files, when no other method of the
source has that name. Named entries run with `dotnet run --project` on the program that runs their test list; attribute
tests (`[Fact]`, `[Theory]`, `[Test]`, `[TestCase]`, `[TestMethod]`) with `dotnet test --filter`. The commands, the
filter variable, the test variables, the tests per method and the smallest body are `CSharpTaskOptions`
(`CSharpTaskFamilies.CreateMethodBody`, `CreateTestSubject`).

**Dependency:** the families read C# with Roslyn, `Microsoft.CodeAnalysis.CSharp` 5.6.0 (pinned), as syntax trees only:
each file is parsed once for both families, with no compilation and no semantic model. This package is the only one
that references it; the library packages (Idrak.Data among them) do not.

License: Apache 2.0.

# Idrak.Text.Plugins

The text model families: decoders, encoders and encoder-decoders, with their GGUF mappings, tool-call formats and RoPE
scalings. The library (`Idrak`, `Idrak.Text`) keeps only generic defaults and fallbacks; the families live here, one
namespace each (`Idrak.Text.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Text.

```csharp
using Idrak.Text.Plugins;

TextPlugins.RegisterAll();      // every family of the package; calling it again changes nothing
```

- `TextPlugins.RegisterAll()` registers every family, `TextPlugins.UnregisterAll()` takes them back (the registries
  are left as they were found), `TextPlugins.All` lists them.
- One family alone: its namespace's static `<Family>Plugin` class, with `Register()` and `Unregister()`.
- Each family is an `IIdrakPlugin` (`Idrak.Abstraction.Plugins`), recorded by name in `IdrakPlugins`; a plug-in
  registered later under the same name takes its place.
- The `idrak` tool registers this package at startup, before any `-P` plug-in (which can still take a family's place);
  `RegisterIdrakPlugin()` is the `-P` entry, for a host that loads `Idrak.Text.Plugins.dll` itself.

A model whose family is not registered is refused with "not registered", naming this package.

## Families

### BERT (`Idrak.Text.Plugins.Bert`)

`BertPlugin.Register()` registers `BertModel`, `BertForMaskedLM` and `BertForSequenceClassification` (config.json's
`architectures`) in `PretrainedEncoders`; `PretrainedEncoderModel.Load` then reads a Hugging Face folder (safetensors
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

BertPlugin.Register();
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

`T5Plugin.Register()` registers `T5ForConditionalGeneration` and `MT5ForConditionalGeneration` in `PretrainedSeq2Seq`,
and `T5EncoderModel` and `MT5EncoderModel` in `PretrainedEncoders`. `PretrainedSeq2SeqModel.Load` then reads a Hugging
Face folder: the shared token embedding (the encoder's input, the decoder's input and, tied, the head's table, one copy),
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

T5Plugin.Register();
using var t5 = PretrainedSeq2SeqModel.Load(folder);
var generator = t5.CreateGenerator();
var (text, reason, stats) = generator.Generate("translate English to German: The house is wonderful.",
    new GenerationOptions { NumBeams = 4, NumPredict = 64 });          // greedy: TopK = 1, RepeatPenalty = 1
```

In the inference engine, an encoder-decoder is the `seq2seq` kind (`engine.Seq2SeqModel(name, folder)`, reached as
`ITextModel`); `idrak run FOLDER "prompt"` answers with it (`--beams N` for beam search). Other `feed_forward_proj`
values, bucket settings that do not fit and checkpoints without a tensor the configuration names are refused naming them.

### Gemma (`Idrak.Text.Plugins.Gemma`)

`GemmaPlugin.Register()` registers `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM` and the text decoder of
`Gemma3ForConditionalGeneration` in `PretrainedArchitectures` (`GemmaArchitectures`); `PretrainedModel.Load` then reads
them on the library's decoder (`DecoderSpec`): gains stored as a difference from 1, embeddings scaled by √width and
tied, and for Gemma 2 and 3 the norms after attention and the feed-forward block too (`GemmaArchitectures.TensorName`),
scores scaled by `query_pre_attn_scalar`, windowed layers alternating as the configuration or `layer_types` says,
soft-capping (Gemma 2), q/k norms and the windowed layers' own rotary base (Gemma 3, transformers 5's
`rope_parameters` too). Gemma writes its configuration from a spec (`PretrainedModel.Create`); Gemma 2 and 3 are read
from checkpoints only. The text decoder of Gemma 3 with images reads `text_config` (with transformers' defaults for
absent keys), never soft-caps its logits, and fits each checkpoint's naming (`language_model.model.` or
`model.language_model.`); its vision part is the Gemma 3 vision plug-in of `Idrak.Vision.Plugins`, which registers this
family when it is missing. Without the vision plug-in such a checkpoint is refused, naming the vision family.

It also registers llama.cpp's `gemma`, `gemma2` and `gemma3` in `GgufArchitectures` (`GemmaGguf`), read as
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

GemmaPlugin.Register();
using var model = PretrainedModel.Load(folder);
```

### Llama (`Idrak.Text.Plugins.Llama`)

`LlamaPlugin.Register()` registers `LlamaForCausalLM` in `PretrainedArchitectures` (`LlamaArchitectures.Llama()`: the
library's common configuration keys and checkpoint names, `PretrainedFamilies.CommonArchitecture`), llama.cpp's `llama`
in `GgufArchitectures` (each head's query and key rows put back in order, experts read as `MixtralForCausalLM`, and Llama
3's RoPE scaling recovered from `rope_freqs.weight`), the GGUF pre-tokenizer names `llama3`, `llama-bpe`, `llama-v3`,
`smaug-bpe`, `falcon3` and `pixtral` in `GgufPreTokenizers` (and `llama3` as `GgufPreTokenizers.Fallback`, the rule
of a name nobody registered, when none is set), the RoPE scaling `llama3` in `RopeScalings` (`LlamaRopeScaling`), and
Llama 3's chat template for a GGUF file without one whose vocabulary holds `<|start_header_id|>`, `llama3` in
`GgufChatTemplates` (after the library's ChatML).

### Qwen (`Idrak.Text.Plugins.Qwen`)

`QwenPlugin.Register()` registers `Qwen2ForCausalLM` (query, key and value biases), `Qwen3ForCausalLM` (q/k norms),
`Qwen2MoeForCausalLM` and `Qwen3MoeForCausalLM` (experts, a gated shared expert; `QwenArchitectures.MoeTensorName`) in
`PretrainedArchitectures`, llama.cpp's `qwen2`, `qwen3`, `qwen2moe` and `qwen3moe` in `GgufArchitectures`, the GGUF
pre-tokenizer names `qwen2`, `qwen35`, `megrez` and `hunyuan`, and the XML tool-call format `qwen3-coder` in
`ToolCallFormats`. The vision-language Qwen models are `Idrak.Vision.Plugins`'.

### Mistral (`Idrak.Text.Plugins.Mistral`)

`MistralPlugin.Register()` registers `MistralForCausalLM` and `MixtralForCausalLM` (experts under `block_sparse_moe`,
`MistralArchitectures.MixtralTensorName`) in `PretrainedArchitectures`, the GGUF pre-tokenizer `tekken`, and the
tool-call format `mistral` (`[TOOL_CALLS]`). Mistral's GGUF files are llama.cpp's `llama`, so it registers the Llama
plug-in when that is missing.

### DeepSeek and harmony (`Idrak.Text.Plugins.DeepSeek`, `Idrak.Text.Plugins.Harmony`)

`DeepSeekPlugin.Register()` registers DeepSeek's special-token tool-call format `deepseek` and the GGUF pre-tokenizer
`deepseek-r1-qwen` (Qwen2's rule); `HarmonyPlugin.Register()` registers the harmony tool-call format `harmony`
(reasoning in the analysis channel, calls as messages to `functions.name`). The tool-call parsers build on the library's
`MarkedToolCallParser` and `ToolCallText`.

```csharp
using Idrak.Models;
using Idrak.Text.Plugins.Qwen;

QwenPlugin.Register();
using var model = PretrainedModel.Load(folder);              // or GgufModel.Prepare("model.gguf") first
```

## Coding languages

The library's coding agent and tools (`CodingAgent`, `CodingTools`, Idrak.Text) name no programming language: they
compose their defaults from the coding languages registered in `CodingLanguages` (`ICodingLanguage`:
`Idrak.Text.Abstractions`). This package has the library's own, each a plug-in in a namespace of its own:

| Namespace | Plug-in | Languages | What they add |
|---|---|---|---|
| `Idrak.Text.Plugins.CSharp` | `CSharpLanguagePlugin` ("csharp-language") | `csharp` (a `*.sln`, `*.slnx` or `*.csproj`) | `dotnet`; checked with `dotnet build`, `dotnet test`; `bin`, `obj`, `.vs` skipped; tasks without commands run `dotnet restore`, `dotnet build --no-restore`, `dotnet test --no-build` |
| `Idrak.Text.Plugins.TypeScript` | `TypeScriptLanguagePlugin` ("typescript-language") | `typescript` (`package.json`, `tsconfig.json`), `angular` (`angular.json`) | `npm`, `npx`, `node`, `tsc` (and `ng`); checked with `npx tsc --noEmit`, `npm test` (`npx ng build`, `npx ng test --watch=false`); `node_modules` linked into a task's workspace, `dist` and `.angular` skipped |
| `Idrak.Text.Plugins.Python` | `PythonLanguagePlugin` ("python-language") | `python` (`pyproject.toml`, `setup.py`, `requirements.txt`, …) | `__pycache__` and `.venv` skipped (no commands) |

With all three registered and every language chosen (`CodingToolOptions.Languages`, `IDRAK_CODING_LANGUAGES`: `all`, the
default), the agent's prompt, its programs and the tools' descriptions are what the library had before the languages
moved out; `detected` keeps those the workspace shows. A workspace no chosen language detects gets a notice
(`CodingTools.LanguageNotice`) naming the plug-in to register.

License: Apache 2.0.

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

License: Apache 2.0.

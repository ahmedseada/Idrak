// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The model-loading built-ins Idrak registers (from <c>LibraryRegistrations</c>, each when its registry is first used): the checkpoint formats (safetensors,
/// PyTorch zip checkpoints read without running them, GGUF), the model sources (folder, local model store, .gguf file), the ggml tensor types, the GGUF
/// pre-tokenizer patterns that are no family's (GPT-2's own rule), the GGUF fallback chat template that is no family's (ChatML), and the tokenizer models (BPE) and tokenizer file readers (tokenizer.json, sentencepiece). Idrak.Data adds the Hugging Face model source.
/// The model families (pretrained and GGUF architectures, their pre-tokenizers and fallback chat templates) are plug-ins' (Idrak.Text.Plugins).
/// </summary>
internal static class LibraryModelFormats
{
    // Both registries ask the most recently registered first: registered in reverse, they are asked GGUF (whose prepared
    // folders are model folders too), then a PyTorch checkpoint (a folder with one and no safetensors), before safetensors
    // (any folder), and folder, store, then gguf.
    public static void RegisterCheckpointFormats()
    {
        CheckpointFormats.Register(new SafeTensorsCheckpointFormat());
        CheckpointFormats.Register(new PyTorchCheckpointFormat());
        CheckpointFormats.Register(new GgufCheckpointFormat());
    }

    public static void RegisterModelSources()
    {
        foreach (var source in Enumerable.Reverse(ModelSource.BuiltIn))
        {
            ModelSources.Register(source);
        }
    }

    public static void RegisterGgufTypes()
    {
        foreach (var (id, type) in GgufFile.BuiltInTypes())
        {
            GgufTypes.Register(id, type);
        }
    }

    public static void RegisterTokenizerModels()
    {
        TokenizerModels.Register(new ModelKind(TokenizerModels.Bpe, "byte-pair encoding: byte-level, and SentencePiece-style with byte fallback", BpeTokenizer.FromDefinition));
        TokenizerModels.Register(new ModelKind(TokenizerModels.WordPiece, "WordPiece (BERT)", WordPieceTokenizer.FromDefinition));
        TokenizerModels.Register(new ModelKind(UnigramTokenizer.Type, "SentencePiece's unigram language model", UnigramTokenizer.FromDefinition));
    }

    // A built-in tokenizer model, registered under its model type.
    private sealed class ModelKind(string name, string summary, Func<TokenizerDefinition, TokenizerOptions, IPretrainedTokenizer> build) : ITokenizerModelKind
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public IPretrainedTokenizer Create(TokenizerModelRequest request) => build(request.Definition, request.Options);
    }

    // Asked in this order: a folder with several files reads tokenizer.json first, then a SentencePiece model, then vocab.txt.
    public static void RegisterTokenizerFiles()
    {
        TokenizerFiles.Register(new TokenizerJsonReader());
        TokenizerFiles.Register(new SentencePieceReader());
        TokenizerFiles.Register(new VocabTxtReader());
    }

    public static void RegisterGgufTokenizers()
    {
        GgufTokenizers.Register(new GgufTokenizerFormats.Converter("gpt2", "byte-level BPE (Llama 3, Qwen, Mistral Nemo, …)", GgufTokenizerFormats.Gpt2));
        GgufTokenizers.Register(new GgufTokenizerFormats.Converter("llama", "SentencePiece BPE (Llama 2, Mistral 7B, TinyLlama, …)", GgufTokenizerFormats.Llama));
        GgufTokenizers.Register(new GgufTokenizerFormats.Converter("t5", "SentencePiece Unigram", GgufTokenizerFormats.T5));
        GgufTokenizers.Register(new GgufTokenizerFormats.Converter("bert", "WordPiece", GgufTokenizerFormats.Bert));
    }

    // ChatML: the turn layout Idrak's own created models use (ModelFiles.ChatMLWithParts) and many families share, so no
    // family's; for a file whose vocabulary holds its start token, as GgufModel chose it before the registry.
    public static void RegisterGgufChatTemplates() =>
        GgufChatTemplates.Register(GgufModel.ChatMLName, new GgufChatTemplate { Matches = (_, tokens) => tokens.Contains("<|im_start|>"), Template = GgufModel.ChatML });

    public static void RegisterGgufPreTokenizers()
    {
        foreach (var (name, pattern) in GgufBuiltIns.PreTokenizers())
        {
            GgufPreTokenizers.Register(name, pattern);
        }
    }
}

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The model-loading built-ins Idrak registers (from <c>LibraryRegistrations</c>, each when its registry is first used): the checkpoint formats (safetensors,
/// PyTorch zip checkpoints read without running them, GGUF), the model sources (folder, local model store, .gguf file), the ggml tensor types, the GGUF families and
/// pre-tokenizer patterns, the pretrained model families, and the tokenizer models (BPE) and tokenizer file readers (tokenizer.json, sentencepiece). Idrak.Data adds the Hugging Face model source.
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

    public static void RegisterGgufArchitectures()
    {
        foreach (var (name, architecture) in GgufBuiltIns.Architectures())
        {
            GgufArchitectures.Register(name, architecture);
        }
    }

    public static void RegisterTokenizerModels()
    {
        TokenizerModels.Register(TokenizerModels.Bpe, BpeTokenizer.FromDefinition);
        TokenizerModels.Register(UnigramTokenizer.Type, UnigramTokenizer.FromDefinition);
    }

    public static void RegisterTokenizerFiles()
    {
        TokenizerFiles.Register(new TokenizerJsonReader());
        TokenizerFiles.Register(new SentencePieceReader());      // after tokenizer.json: a folder with both reads that
    }

    public static void RegisterGgufPreTokenizers()
    {
        foreach (var (name, pattern) in GgufBuiltIns.PreTokenizers())
        {
            GgufPreTokenizers.Register(name, pattern);
        }
    }

    public static void RegisterPretrainedArchitectures()
    {
        foreach (var (name, architecture) in PretrainedFamilies.BuiltIns())
        {
            PretrainedArchitectures.Register(name, architecture);
        }
    }
}

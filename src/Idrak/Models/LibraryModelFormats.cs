// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>
/// The model-loading built-ins Idrak registers (from <c>LibraryRegistrations</c>, each when its registry is first used): the checkpoint formats (safetensors,
/// GGUF), the model sources (folder, local model store, .gguf file), the ggml tensor types, the GGUF families and
/// pre-tokenizer patterns, and the pretrained model families. Idrak.Data adds the Hugging Face model source.
/// </summary>
internal static class LibraryModelFormats
{
    // Both registries ask the most recently registered first: registered in reverse, they are asked GGUF (whose prepared
    // folders are model folders too) before safetensors, and folder, store, then gguf.
    public static void RegisterCheckpointFormats()
    {
        CheckpointFormats.Register(new SafeTensorsCheckpointFormat());
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

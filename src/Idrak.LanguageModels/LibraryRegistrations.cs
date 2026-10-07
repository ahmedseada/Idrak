// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.LanguageModels;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the checkpoint formats
/// (safetensors, GGUF), the model sources (folder, local model store, .gguf file, Hugging Face id), the ggml tensor types,
/// the GGUF families and pre-tokenizer patterns, and the pretrained model families. <c>LibraryDefaults.Ensure</c> (in
/// Idrak.Abstraction) calls <see cref="RegisterAll"/> once, before any of those registries is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static void RegisterAll()
    {
        // Both registries ask the most recently registered first: registered in reverse, they are asked GGUF (whose prepared
        // folders are model folders too) before safetensors, and folder, store, gguf, then a Hugging Face id.
        CheckpointFormats.Register(new SafeTensorsCheckpointFormat());
        CheckpointFormats.Register(new GgufCheckpointFormat());
        foreach (var source in Enumerable.Reverse(ModelSource.BuiltIn))
        {
            ModelSources.Register(source);
        }

        foreach (var (id, type) in GgufFile.BuiltInTypes())
        {
            GgufTypes.Register(id, type);
        }

        foreach (var (name, architecture) in GgufBuiltIns.Architectures())
        {
            GgufArchitectures.Register(name, architecture);
        }

        foreach (var (name, pattern) in GgufBuiltIns.PreTokenizers())
        {
            GgufPreTokenizers.Register(name, pattern);
        }

        foreach (var (name, architecture) in PretrainedFamilies.BuiltIns())
        {
            PretrainedArchitectures.Register(name, architecture);
        }
    }
}

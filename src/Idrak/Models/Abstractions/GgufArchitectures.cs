// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// How llama.cpp stores one model family (general.architecture in the file): the Hugging Face architecture it corresponds
/// to, and whether it interleaved the query and key rows of each head for its rotary layout. The file's settings are read
/// from the usual keys under the family's name ({name}.embedding_length, {name}.block_count, …) and its tensors from
/// llama.cpp's usual names (token_embd, blk.N.attn_q, …). Register new families with <see cref="GgufArchitectures.Register"/>.
/// </summary>
public sealed class GgufArchitecture
{
    /// <summary>The Hugging Face architecture (config.json's "architectures"), registered in <see cref="PretrainedArchitectures"/>.</summary>
    public required string HuggingFace { get; init; }

    /// <summary>Whether llama.cpp interleaved each head's query and key rows (Llama does; they are put back on load).</summary>
    public bool InterleavedQueryKeys { get; init; }

    /// <summary>
    /// The Hugging Face architecture of this family's files with experts ({name}.expert_count above 0), when it is not
    /// <see cref="HuggingFace"/> (llama files with experts are Mixtral); null: <see cref="HuggingFace"/>.
    /// </summary>
    public string? WithExperts { get; init; }

    /// <summary>
    /// Whether the family renormalizes the chosen experts' weights when the file does not say ({name}.expert_weights_norm),
    /// as llama.cpp decides per family; null leaves it to the Hugging Face family's default.
    /// </summary>
    public bool? NormalizeTopK { get; init; }
}

/// <summary>
/// The model families GGUF files are read as, by GGUF architecture name. Idrak registers llama (Llama,
/// Mistral, and Mixtral when the file has experts), qwen2, qwen3, qwen2moe and qwen3moe; add others with
/// <see cref="Register"/>.
/// </summary>
public static class GgufArchitectures
{
    private static readonly Dictionary<string, GgufArchitecture> Registry = new(StringComparer.Ordinal);

    static GgufArchitectures() => LibraryModelFormats.RegisterGgufArchitectures();   // the built-in families, on first use

    /// <summary>Registers (or replaces) how to read the GGUF architecture <paramref name="name"/>.</summary>
    public static void Register(string name, GgufArchitecture architecture)
    {
        ArgumentNullException.ThrowIfNull(architecture);
        lock (Registry)
        {
            Registry[name] = architecture;
        }
    }

    /// <summary>Removes the architecture registered as <paramref name="name"/>; false when there is none.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>The architecture registered as <paramref name="name"/>.</summary>
    public static GgufArchitecture Get(string name) => Find(name)
        ?? throw new NotSupportedException($"No GGUF architecture '{name}' is registered ({string.Join(", ", Names)}); add it with GgufArchitectures.Register.");

    /// <summary>The architecture registered as <paramref name="name"/>, or null when there is none.</summary>
    public static GgufArchitecture? Find(string name)
    {
        lock (Registry)
        {
            return Registry.GetValueOrDefault(name);
        }
    }
}

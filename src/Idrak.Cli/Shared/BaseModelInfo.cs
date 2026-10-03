// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.LanguageModels;

namespace Idrak.Cli.Shared;

/// <summary>
/// What the design rules need of a base model (<c>idrak suggest --base MODEL</c>) without loading its weights: the
/// parameter count, sizes and context, and its tokenizer to count tokens. The model is found as every model command
/// finds it without downloading (<see cref="ModelCache.TryLocate"/>: a folder, a .gguf file, a cached Hugging Face id
/// or pulled GGUF file under <c>--cache</c>, or Hugging Face's own cache) and read with <see cref="ModelFacts"/>; when
/// it is not there the rules continue with what they know and say so (<c>idrak pull</c> fetches it).
/// </summary>
internal sealed record BaseModelInfo(string Name, string? Folder, long? Parameters, int? Hidden, int? Layers, int? Context, BpeTokenizer? Tokenizer)
{
    /// <summary>Reads what is available about <paramref name="name"/> (after alias resolution).</summary>
    public static BaseModelInfo Inspect(CommandContext context, string name)
    {
        var unknown = new BaseModelInfo(name, null, null, null, null, null, null);
        ModelCache.Local? local;
        ModelFacts facts;
        try
        {
            local = ModelCache.TryLocate(context, name);
            if (local is null)
            {
                return unknown;
            }

            facts = ModelFacts.Read(local);
        }
        catch (Exception e) when (e is DirectoryNotFoundException or FileNotFoundException or InvalidOperationException or InvalidDataException
                                      or NotSupportedException or UsageException or System.Text.Json.JsonException)
        {
            return unknown;
        }

        // The decoder's sizes when the family is registered, else the config's own fields.
        var text = facts.Config["text_config"] as JsonObject ?? facts.Config;
        var spec = facts.Spec;
        string tokenizer = Path.Combine(local.Folder, "tokenizer.json");
        return new BaseModelInfo(name, local.Folder, facts.Parameters > 0 ? facts.Parameters : null,
            spec?.Dim ?? (int?)text["hidden_size"], spec?.Layers ?? (int?)text["num_hidden_layers"], spec?.MaxPositions ?? (int?)text["max_position_embeddings"],
            File.Exists(tokenizer) ? BpeTokenizer.Load(tokenizer) : null);
    }
}

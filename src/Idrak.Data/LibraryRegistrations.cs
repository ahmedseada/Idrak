// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the Hugging Face model source.
/// <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterFor"/> once per registry, before that
/// registry is first used. The registries that live here (data file formats, dataset sources, Parquet codecs) register
/// their built-ins themselves.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(ModelSources)] = () => ModelSources.Register(HuggingFaceModels.Source),
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}

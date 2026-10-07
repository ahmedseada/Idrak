// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: its data file formats (JSON
/// Lines, JSON, CSV, TSV, Parquet, text, source code), its dataset sources (hf, github, kaggle, zenodo, http, folder,
/// file), its Parquet codecs and the Hugging Face model source. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DataFileFormats)] = DataFiles.RegisterAll,
        [typeof(DatasetSources)] = LibraryDatasetSources.RegisterAll,
        [typeof(ParquetCodecs)] = Parquet.Codecs.RegisterAll,
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

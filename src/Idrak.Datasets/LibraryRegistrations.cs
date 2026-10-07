// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Datasets;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: its data file formats (JSON
/// Lines, JSON, CSV, TSV, Parquet, text, source code), its dataset sources (hf, github, kaggle, zenodo, http, folder,
/// file) and its Parquet codecs. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterAll"/>
/// once, before any of those registries is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static void RegisterAll()
    {
        DataFiles.RegisterAll();
        LibraryDatasetSources.RegisterAll();
        Parquet.Codecs.RegisterAll();
    }
}

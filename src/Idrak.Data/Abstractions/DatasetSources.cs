// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Data.Abstractions;

/// <summary>
/// The rows a <see cref="IDatasetSource"/> opened (JSON objects, column name → value), read lazily: each enumeration reads
/// the source again. <c>DatasetRows</c> (Idrak.Data) implements it and wraps any other implementation, so the filters,
/// mixes and splits apply to rows from every source.
/// </summary>
public interface IDatasetRows : IEnumerable<JsonObject>
{
    /// <summary>A name for messages and reports (the source the rows are read from).</summary>
    string Name { get; }

    /// <summary>
    /// The local files the rows are read from, downloading remote ones into the cache first (without reading rows); throws
    /// <see cref="InvalidOperationException"/> for rows that are not read from files of their own (derived from other rows).
    /// </summary>
    IReadOnlyList<string> Download();
}

/// <summary>
/// A kind of <see cref="DatasetSpec"/> source (such as <c>hf:org/name</c> or a local folder): which sources it opens and how
/// they become rows.
/// </summary>
public interface IDatasetSource
{
    /// <summary>The source kind's name (the built-ins: hf, github, kaggle, zenodo, http, folder, file).</summary>
    string Name { get; }

    /// <summary>Options this source reads beyond the ones every source has (<see cref="DatasetSpec.CommonOptions"/>); others are refused.</summary>
    IReadOnlyCollection<string> Options => [];

    /// <summary>Whether this kind opens <paramref name="source"/> (the spec's source without options), usually by its prefix.</summary>
    bool CanOpen(string source);

    /// <summary>
    /// The rows of <paramref name="spec"/>, read with <paramref name="options"/> (built from the spec's options);
    /// <paramref name="downloader"/> fetches remote files (null: the library's shared one).
    /// </summary>
    IDatasetRows Open(DatasetSpec spec, ReadOptions options, IDownloader? downloader);
}


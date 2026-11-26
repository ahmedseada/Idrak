// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Data.Abstractions;

/// <summary>What a data file format's files hold (<see cref="IDataFileFormat.Kind"/>), which decides how reading treats them.</summary>
public enum DataFileKind
{
    /// <summary>Rows written as text (JSON Lines, CSV, plain text): read as rows, or as one document per file when documents are asked for.</summary>
    Table,

    /// <summary>Rows in a binary layout (Parquet): read as rows only, never as a document.</summary>
    Binary,

    /// <summary>Source code: one row per file, chosen by extension only when code is asked for.</summary>
    Code,
}

/// <summary>
/// A kind of data file a dataset reads (<c>DatasetRows</c> and <c>DataFiles</c> in Idrak.Data): its name, the extensions it is chosen by, what its files hold
/// (<see cref="Kind"/>) and how its rows are read. A reading names one with <see cref="ReadOptions.FileFormat"/>.
/// </summary>
public interface IDataFileFormat
{
    /// <summary>The format's name (the built-ins use the names of <see cref="DataFormat"/>, e.g. "Csv").</summary>
    string Name { get; }

    /// <summary>The extensions files of this format have, with the dot (".csv"); a trailing ".gz" is stripped before matching.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>What its files hold: rows as text (<see cref="DataFileKind.Table"/>, unless the format says otherwise), rows in a binary layout, or source code.</summary>
    DataFileKind Kind => DataFileKind.Table;

    /// <summary>The rows of a file of this format; <paramref name="open"/> opens it (already decompressed), <paramref name="path"/> is shown in rows and errors.</summary>
    IEnumerable<JsonObject> Read(Func<Stream> open, string path, ReadOptions options);
}


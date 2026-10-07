// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>The built-in data file formats (registered in <see cref="DataFileFormats"/> under these names by Idrak.Datasets).</summary>
public enum DataFormat
{
    /// <summary>One JSON object per line (.jsonl, .ndjson).</summary>
    JsonLines,

    /// <summary>A JSON document (.json): an array of objects, an object holding such an array, or columns of equal length.</summary>
    Json,

    /// <summary>Comma-separated values with a header row (.csv).</summary>
    Csv,

    /// <summary>Tab-separated values with a header row (.tsv).</summary>
    Tsv,

    /// <summary>Apache Parquet (.parquet), the format of most Hugging Face datasets.</summary>
    Parquet,

    /// <summary>Plain text (.txt, .md): see <see cref="ReadOptions.Text"/>.</summary>
    Text,

    /// <summary>A source file, as one row {"text", "path", "language"}.</summary>
    Code,
}

/// <summary>How text files become rows.</summary>
public enum TextRows
{
    /// <summary>Each non-empty line is a row {"text"} (as Hugging Face's text loader).</summary>
    Lines,

    /// <summary>Each block of text between blank lines is a row.</summary>
    Paragraphs,

    /// <summary>The whole file is one row {"text", "path"}.</summary>
    Document,
}

/// <summary>How a dataset reads files (<c>Dataset</c> and <c>DataFiles</c> in Idrak.Datasets, and every <see cref="IDataFileFormat"/>).</summary>
public sealed record ReadOptions
{
    /// <summary>The defaults.</summary>
    public static ReadOptions Default { get; } = new();

    /// <summary>The format, instead of choosing it by extension.</summary>
    public DataFormat? Format { get; init; }

    /// <summary>
    /// The format as an <see cref="IDataFileFormat"/> (for example one added with <see cref="DataFileFormats.Register"/>),
    /// instead of choosing it by extension; takes precedence over <see cref="Format"/>.
    /// </summary>
    public IDataFileFormat? FileFormat { get; init; }

    /// <summary>Text files: a row per line (default), paragraph or file.</summary>
    public TextRows Text { get; init; } = TextRows.Lines;

    /// <summary>CSV / TSV: turn numbers and true/false into JSON numbers and booleans, empty cells into null.</summary>
    public bool InferTypes { get; init; } = true;

    /// <summary>JSON documents: the property holding the rows (default: the only array of objects, if there is one).</summary>
    public string? JsonProperty { get; init; }

    /// <summary>Only files (and archive entries) whose path matches this glob, e.g. <c>*.cs</c> or <c>data/train-*.parquet</c>.</summary>
    public string? Pattern { get; init; }

    /// <summary>Add a "_file" column with the path each row came from.</summary>
    public bool IncludeFile { get; init; }

    /// <summary>Also read source code files (as <see cref="DataFormat.Code"/>) in folders and archives.</summary>
    public bool IncludeCode { get; init; }

    /// <summary>
    /// Every text file (data files such as .json and .csv included) is one row {"text", "path", "language"}: a repository
    /// or folder read as documents rather than as data. Implies <see cref="IncludeCode"/>.
    /// </summary>
    public bool Documents { get; init; }

    /// <summary>Largest source or text file read as one document, in bytes (larger ones, often generated, are skipped).</summary>
    public long MaxDocumentBytes { get; init; } = 1 << 20;

    /// <summary>The folder paths are shown relative to (set by <c>Dataset.FromFolder</c>).</summary>
    public string? Root { get; init; }
}

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Data.Abstractions;

namespace Idrak.Data.Code;

/// <summary>
/// The library's language-neutral family "file-text" (plan 36): one pretraining row per source file, <c>{"text", "path",
/// "language"}</c> as Idrak.Data's <c>code</c> format reads a repository, for every file whose extension names a language
/// (<see cref="DataFiles.LanguageOf"/>). It proposes no tasks. Duplicates are the row filters' to drop (<see cref="RowFilters"/>),
/// as the extractor passes every family's rows through them.
/// </summary>
public sealed class FileTextFamily : ITaskFamily
{
    /// <summary>Its name in <see cref="TaskFamilies"/>: "file-text".</summary>
    public const string FamilyName = "file-text";

    /// <inheritdoc/>
    public string Name => FamilyName;

    /// <inheritdoc/>
    public string Summary => "one pretraining row per source file ({text, path, language}), through the row filters; no tasks";

    /// <inheritdoc/>
    public async IAsyncEnumerable<TaskDraft> DraftAsync(CodeSource source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await Task.CompletedTask.ConfigureAwait(false);
        yield break;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<JsonObject> RowsAsync(CodeSource source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        var files = source.Files.Count > 0 ? source.Files : CodeExtractor.Walk(source);
        foreach (var file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (file.Language is null)
            {
                continue;
            }

            byte[] bytes = await File.ReadAllBytesAsync(Path.Combine(source.Root, file.Path), cancellationToken).ConfigureAwait(false);
            var text = bytes.AsSpan();
            if (text.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
            {
                text = text[3..];                                           // the UTF-8 byte order mark is not text
            }

            yield return new JsonObject { ["text"] = Encoding.UTF8.GetString(text), ["path"] = file.Path, ["language"] = file.Language };
        }
    }
}

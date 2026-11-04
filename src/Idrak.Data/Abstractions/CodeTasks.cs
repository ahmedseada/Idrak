// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Data.Abstractions;

/// <summary>
/// A repository or project a code extractor reads (plan 36): its root folder, the coding languages it is read in, its
/// history when it has one, and how its files are read. The languages are the names of coding languages registered with
/// Idrak.Text's <c>CodingLanguages</c> (this package cannot name them): nothing here assumes one.
/// </summary>
/// <param name="Root">The repository's or project's root folder.</param>
public sealed record CodeSource(string Root)
{
    /// <summary>The coding languages it is read in, by their registered names; empty: those that detect it.</summary>
    public IReadOnlyList<string> Languages { get; init; } = [];

    /// <summary>Its history (<see cref="RepositoryHistories"/>), or null when it has none or none is read.</summary>
    public IRepositoryHistory? History { get; init; }

    /// <summary>How its files are read (skipped folders, the largest file).</summary>
    public ReadOptions ReadOptions { get; init; } = ReadOptions.Default;

    /// <summary>
    /// The files the extractor walked once for every family (rule 72: the tree walked once), in the order walked; empty: a
    /// family walks <see cref="Root"/> itself (<c>CodeExtractor.Walk</c> in Idrak.Data walks as the extractor does).
    /// </summary>
    public IReadOnlyList<CodeSourceFile> Files { get; init; } = [];
}

/// <summary>A file of a <see cref="CodeSource"/>: where it is, what it is in, and its size; its bytes are read by whoever needs them.</summary>
/// <param name="Path">The path relative to the root, with '/'.</param>
/// <param name="Language">The language its extension names (<c>DataFiles.LanguageOf</c>), or null for none.</param>
/// <param name="Length">Its size in bytes.</param>
public sealed record CodeSourceFile(string Path, string? Language, long Length);

/// <summary>
/// A file a task writes, by its path relative to the task's workspace, as bytes (rule 74: read and written as bytes, never
/// decoded on the way); <see cref="Delete"/> removes the path instead.
/// </summary>
/// <param name="Path">The path relative to the workspace, with '/'.</param>
/// <param name="Bytes">The file's bytes (empty when <see cref="Delete"/>).</param>
/// <param name="Delete">Whether the path is removed rather than written.</param>
public sealed record CodeFile(string Path, ReadOnlyMemory<byte> Bytes, bool Delete = false)
{
    /// <summary>A removal of <paramref name="path"/>.</summary>
    public static CodeFile Removed(string path) => new(path, ReadOnlyMemory<byte>.Empty, Delete: true);
}

/// <summary>Where a task came from, for the manifest and for leakage checks between splits.</summary>
public sealed record TaskProvenance
{
    /// <summary>The source file the task was made from, relative to the root, or null.</summary>
    public string? Path { get; init; }

    /// <summary>The commit the task was made from (<see cref="CodeCommit.Id"/>), or null.</summary>
    public string? Commit { get; init; }

    /// <summary>The test the task is checked by (its full name, as the test runner names it), or null.</summary>
    public string? Test { get; init; }

    /// <summary>The group the task belongs to for splits (a test group, a project, a contract, a date range), or null.</summary>
    public string? Group { get; init; }
}

/// <summary>
/// A task a <see cref="ITaskFamily"/> proposes, not yet proven: the prompt, the files that make its workspace from the
/// source, the hidden files only the check sees, the setup and check commands, the files of a solution, its tags and where
/// it came from. A verifier keeps it only when the check fails on the workspace and passes with the solution.
/// </summary>
/// <param name="Id">A stable id (the same source gives the same id).</param>
/// <param name="Family">The family that made it (<see cref="IKind.Name"/>).</param>
/// <param name="Prompt">What the task asks, in words.</param>
public sealed record TaskDraft(string Id, string Family, string Prompt)
{
    /// <summary>The coding language it is in (a registered name), or null for the source's.</summary>
    public string? Language { get; init; }

    /// <summary>The files written over a copy of the source to make the workspace (a method removed, a commit reverted).</summary>
    public IReadOnlyList<CodeFile> Workspace { get; init; } = [];

    /// <summary>The hidden files the check adds (tests the workspace does not show).</summary>
    public IReadOnlyList<CodeFile> Verify { get; init; } = [];

    /// <summary>The files of a solution, written over the workspace (the source as it was, by default).</summary>
    public IReadOnlyList<CodeFile> Solution { get; init; } = [];

    /// <summary>Commands run once before the check (a restore), as command lines.</summary>
    public IReadOnlyList<string> Setup { get; init; } = [];

    /// <summary>The check's commands, in order: all exit with 0 when the task is solved.</summary>
    public IReadOnlyList<string> VerifyCommands { get; init; } = [];

    /// <summary>Variables the setup and check commands run with (a test runner's filter), over the tools' own; empty for none.</summary>
    public IReadOnlyDictionary<string, string> Environment { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Tags (the family's kind of task, "weak" for a pair that is not proven by a check).</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>Where it came from.</summary>
    public TaskProvenance Provenance { get; init; } = new();
}

/// <summary>How a changed file of a commit changed.</summary>
public enum CodeChangeKind
{
    /// <summary>The commit adds the file.</summary>
    Added,

    /// <summary>The commit changes the file.</summary>
    Modified,

    /// <summary>The commit removes the file.</summary>
    Deleted,

    /// <summary>The commit moves the file (<see cref="CodeChange.OldPath"/> is where it was).</summary>
    Renamed,
}

/// <summary>A file a commit changes.</summary>
/// <param name="Path">Its path after the commit, relative to the root, with '/'.</param>
/// <param name="Kind">How it changed.</param>
/// <param name="OldPath">Its path before the commit, for a rename; null otherwise.</param>
public readonly record struct CodeChange(string Path, CodeChangeKind Kind, string? OldPath = null);

/// <summary>
/// A commit of a <see cref="IRepositoryHistory"/>: its id, message, author and author date, its parents and the files it
/// changes. Its diff is not held here: <see cref="IRepositoryHistory.OpenDiffAsync"/> streams it.
/// </summary>
/// <param name="Id">The commit's id (a hash).</param>
/// <param name="Message">Its whole message.</param>
/// <param name="AuthorDate">When it was authored.</param>
public sealed record CodeCommit(string Id, string Message, DateTimeOffset AuthorDate)
{
    /// <summary>Its author's name, or null.</summary>
    public string? Author { get; init; }

    /// <summary>Its parents' ids (none for a first commit, two or more for a merge).</summary>
    public IReadOnlyList<string> Parents { get; init; } = [];

    /// <summary>The files it changes.</summary>
    public IReadOnlyList<CodeChange> Changes { get; init; } = [];
}

/// <summary>Which commits a <see cref="IRepositoryHistory"/> reads.</summary>
public sealed record HistoryQuery
{
    /// <summary>The revision to start from (newest first from there): "HEAD" by default.</summary>
    public string From { get; init; } = "HEAD";

    /// <summary>The most commits read; null for all.</summary>
    public int? MaxCount { get; init; }

    /// <summary>Only commits authored at or after this time; null for no bound.</summary>
    public DateTimeOffset? Since { get; init; }

    /// <summary>Only commits authored at or before this time; null for no bound.</summary>
    public DateTimeOffset? Until { get; init; }

    /// <summary>Only commits that change one of these paths (relative to the root); empty for all.</summary>
    public IReadOnlyList<string> Paths { get; init; } = [];

    /// <summary>Whether merge commits are read (false by default: their changes are their parents').</summary>
    public bool Merges { get; init; }
}

/// <summary>
/// A family of coding tasks (<see cref="TaskFamilies"/>): given a <see cref="CodeSource"/>, it proposes <see cref="TaskDraft"/>s,
/// streamed (a repository's drafts are never all held at once). It reads files as bytes and walks them once, in order. It
/// proves nothing itself: a verifier keeps a draft only when its check fails on the workspace and passes with the solution.
/// </summary>
public interface ITaskFamily : IKind
{
    /// <summary>The coding languages it works in (registered names); empty: any.</summary>
    IReadOnlyList<string> Languages => [];

    /// <summary>Whether it needs the source's history (<see cref="CodeSource.History"/>); a source without one gets no drafts from it.</summary>
    bool NeedsHistory => false;

    /// <summary>The drafts it proposes for <paramref name="source"/>, in a stable order.</summary>
    /// <param name="source">The repository or project.</param>
    /// <param name="cancellationToken">Stops the walk (the enumeration throws <see cref="OperationCanceledException"/>).</param>
    IAsyncEnumerable<TaskDraft> DraftAsync(CodeSource source, CancellationToken cancellationToken = default);

    /// <summary>
    /// The rows it writes for pretraining or tuning (JSON objects, such as <c>{"text", "path", "language"}</c>), in a stable
    /// order, streamed; none by default (a family of tasks only). The extractor passes them through the row filters
    /// (<see cref="RowFilters"/>) and writes them under its output's rows folder.
    /// </summary>
    IAsyncEnumerable<JsonObject> RowsAsync(CodeSource source, CancellationToken cancellationToken = default) => CodeRows.None(cancellationToken);
}

/// <summary>Helpers for <see cref="ITaskFamily.RowsAsync"/>.</summary>
public static class CodeRows
{
    /// <summary>No rows: the default of a family of tasks only.</summary>
    public static async IAsyncEnumerable<JsonObject> None([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await Task.CompletedTask.ConfigureAwait(false);
        yield break;
    }
}

/// <summary>
/// The task families by name (ignoring case), for <c>idrak data code --families</c> and <c>CodeExtractorOptions.Families</c>.
/// The library registers its mechanical families in Idrak.Data (plan 36, G1 and G2); an app or a plug-in registers its own
/// with <see cref="Register"/>. A name nobody registered is refused with "not registered", naming the registered ones.
/// </summary>
public static class TaskFamilies
{
    private static readonly SlotTable<string, ITaskFamily> Table = new(nameof(TaskFamilies), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a family's drafts are streamed into a dataset: falling back to another family's would change what the dataset holds");

    // The library's language-neutral families, on first use: file-text (Idrak.Data.Code.FileTextFamily).
    static TaskFamilies() => Overrides.AsLibraryDefaults(() => Register(new Idrak.Data.Code.FileTextFamily()));

    /// <summary>Registers <paramref name="family"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITaskFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrWhiteSpace(family.Name);
        Table.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static ITaskFamily? Find(string name) => Table.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static ITaskFamily Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(TaskFamilies), name, Table.Keys,
            $"Task family '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); add it with TaskFamilies.Register.");

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITaskFamily? Default(string name) => Table.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered family as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(f => $"{f.Name}: {f.Summary}"));
}

/// <summary>
/// A repository's history (<see cref="RepositoryHistories"/>): its commits, newest first, streamed one at a time, and each
/// commit's diff and files at a commit as streams, never loaded whole. The library's (plan 36, G2) runs <c>git</c> as a
/// process through the registered command sandbox; an app can register another reader.
/// </summary>
public interface IRepositoryHistory : IKind
{
    /// <summary>Whether it reads the history of the repository at <paramref name="root"/>.</summary>
    bool CanRead(string root);

    /// <summary>The commits <paramref name="query"/> asks for (all from HEAD when null), newest first, as they are read.</summary>
    IAsyncEnumerable<CodeCommit> CommitsAsync(string root, HistoryQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary>The commit's diff against its first parent (unified, as bytes), as a stream the caller disposes.</summary>
    ValueTask<Stream> OpenDiffAsync(string root, string commit, CancellationToken cancellationToken = default);

    /// <summary>The bytes of <paramref name="path"/> at <paramref name="commit"/> as a stream the caller disposes, or null when it has no such file.</summary>
    ValueTask<Stream?> OpenFileAsync(string root, string commit, string path, CancellationToken cancellationToken = default);
}

/// <summary>
/// The repository-history readers by name (ignoring case). The library registers its git reader in Idrak.Data (plan 36,
/// G2); an app registers another with <see cref="Register"/>. A name nobody registered is refused with "not registered".
/// </summary>
public static class RepositoryHistories
{
    private static readonly SlotTable<string, IRepositoryHistory> Table = new(nameof(RepositoryHistories), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a history is read as streams with effects (a process per read): a read is not retried on another reader");

    /// <summary>Registers <paramref name="history"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IRepositoryHistory history)
    {
        ArgumentNullException.ThrowIfNull(history);
        ArgumentException.ThrowIfNullOrWhiteSpace(history.Name);
        Table.Register(history.Name, history, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's reader <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The reader registered as <paramref name="name"/>, or null.</summary>
    public static IRepositoryHistory? Find(string name) => Table.Find(name);

    /// <summary>The reader registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static IRepositoryHistory Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(RepositoryHistories), name, Table.Keys,
            $"Repository history '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); add it with RepositoryHistories.Register.");

    /// <summary>The first registered reader that reads the repository at <paramref name="root"/>, or null.</summary>
    public static IRepositoryHistory? For(string root) => Table.Values.FirstOrDefault(h => h.CanRead(root));

    /// <summary>The library's reader <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IRepositoryHistory? Default(string name) => Table.Default(name);

    /// <summary>Who registered the reader <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered reader as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(h => $"{h.Name}: {h.Summary}"));
}

/// <summary>
/// What a <see cref="IRowFilter"/> run is given: which column holds the text, the threshold its kind reads (a
/// near-duplicate similarity, a size), and where the rows it drops are told with their reason (a manifest's counts).
/// </summary>
public sealed record RowFilterContext
{
    /// <summary>The column holding the text the filter judges ("text" by default).</summary>
    public string TextColumn { get; init; } = "text";

    /// <summary>The filter's threshold (its own meaning: a similarity, a size), or null for its default.</summary>
    public double? Threshold { get; init; }

    /// <summary>Told each row the filter drops, with the reason in words; null: none told.</summary>
    public Action<JsonObject, string>? Rejected { get; init; }
}

/// <summary>
/// A filter on rows (<see cref="RowFilters"/>): deduplication (exact, near) and quality checks (size, binary content,
/// generated files). It keeps rows in their order and reads them once, lazily; a filter that remembers rows (dedupe) keeps
/// its state per call of <see cref="Apply"/>, so each run starts clean.
/// </summary>
public interface IRowFilter : IKind
{
    /// <summary>The rows of <paramref name="rows"/> it keeps, in order, as they are read; each dropped one is told to <see cref="RowFilterContext.Rejected"/>.</summary>
    IEnumerable<JsonObject> Apply(IEnumerable<JsonObject> rows, RowFilterContext context);
}

/// <summary>
/// The row filters by name (ignoring case). The library registers its own in Idrak.Data (plan 36, G2: exact and near
/// dedupe, size, binary, generated); an app registers another with <see cref="Register"/>. A name nobody registered is
/// refused with "not registered".
/// </summary>
public static class RowFilters
{
    private static readonly SlotTable<string, IRowFilter> Table = new(nameof(RowFilters), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a filter decides which rows a dataset keeps, with state across the rows: it is not swapped for another midway");

    /// <summary>Registers <paramref name="filter"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IRowFilter filter)
    {
        ArgumentNullException.ThrowIfNull(filter);
        ArgumentException.ThrowIfNullOrWhiteSpace(filter.Name);
        Table.Register(filter.Name, filter, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's filter <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The filter registered as <paramref name="name"/>, or null.</summary>
    public static IRowFilter? Find(string name) => Table.Find(name);

    /// <summary>The filter registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names it and the registered ones.</exception>
    public static IRowFilter Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(RowFilters), name, Table.Keys,
            $"Row filter '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); add it with RowFilters.Register.");

    /// <summary>The library's filter <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IRowFilter? Default(string name) => Table.Default(name);

    /// <summary>Who registered the filter <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered filter as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(f => $"{f.Name}: {f.Summary}"));
}

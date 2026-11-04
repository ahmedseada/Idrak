// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Text.Abstractions;

/// <summary>
/// A change of a file the coding tools are about to make for the model (<c>write_file</c> or <c>edit_file</c>), as an
/// <see cref="IWritePolicy"/> sees it before anything is written.
/// </summary>
/// <param name="Tool">The tool that writes: "write_file" or "edit_file".</param>
/// <param name="Path">The file, relative to the workspace (with '/').</param>
/// <param name="FullPath">The file's absolute path.</param>
/// <param name="Creates">Whether the file is new (it does not exist yet).</param>
/// <param name="Content">The file's whole text after the write.</param>
/// <param name="Previous">The file's text before the write; null when it is created.</param>
/// <param name="Change">What changes, in words ("12 lines", "1 replacement"), for a question or a log.</param>
public sealed record FileWrite(string Tool, string Path, string FullPath, bool Creates, string Content, string? Previous, string Change);

/// <summary>
/// Whether a write of the coding tools may happen (<see cref="WritePolicies"/>): the library's <see cref="WritePolicies.Allow"/>
/// lets every write inside the workspace happen; an app registers one that asks its user (<see cref="AskingWritePolicy"/>,
/// <see cref="WritePolicies.Ask"/>), keeps a log, or refuses some files. It is asked once per write, after the tool has
/// found what to write and before it writes; a write it refuses leaves the file as it was, and the model reads the reason.
/// Thread-safe: the tools of several agents may ask at once.
/// </summary>
public interface IWritePolicy
{
    /// <summary>Its name in <see cref="WritePolicies"/> ("allow", "ask").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>Null to let <paramref name="write"/> happen; else why not, in words the model reads ("the user said no").</summary>
    /// <param name="write">The write, before it is made.</param>
    /// <param name="cancellationToken">Cancels the question (the tool call ends with it).</param>
    ValueTask<string?> CheckAsync(FileWrite write, CancellationToken cancellationToken = default);
}

/// <summary>
/// A write policy that asks (an app's user, a reviewer, a rule of its own) through <paramref name="ask"/>: true lets the
/// write happen, false refuses it with <see cref="Refusal"/>. Registered under <see cref="WritePolicies.Ask"/> unless
/// another name is given; the library registers none, since only an app has someone to ask (the CLI's <c>idrak agent</c>
/// registers one that asks on its terminal).
/// </summary>
/// <param name="ask">The question: true to write, false to refuse.</param>
/// <param name="name">Its name in <see cref="WritePolicies"/> (<see cref="WritePolicies.Ask"/> by default).</param>
public sealed class AskingWritePolicy(Func<FileWrite, CancellationToken, ValueTask<bool>> ask, string name = WritePolicies.Ask) : IWritePolicy
{
    private readonly Func<FileWrite, CancellationToken, ValueTask<bool>> _ask = ask ?? throw new ArgumentNullException(nameof(ask));

    /// <inheritdoc />
    public string Name { get; } = string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("A write policy needs a name.", nameof(name)) : name;

    /// <inheritdoc />
    public string Summary => "asks before every write_file and edit_file (y to write)";

    /// <summary>The reason a refused write gives the model ("the user did not allow it" by default).</summary>
    public string Refusal { get; init; } = "the user did not allow it";

    /// <inheritdoc />
    public async ValueTask<string?> CheckAsync(FileWrite write, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(write);
        return await _ask(write, cancellationToken).ConfigureAwait(false) ? null : Refusal;
    }
}

/// <summary>
/// The write policies of the coding tools' <c>write_file</c> and <c>edit_file</c>, by name (ignoring case). The library
/// registers <see cref="Allow"/> (every write inside the workspace happens: the default); an app registers
/// <see cref="Ask"/> (an <see cref="AskingWritePolicy"/> with its own question) or its own. The tools take theirs from
/// their options' <c>WritePolicy</c>, else <see cref="EnvironmentVariable"/> (IDRAK_WRITE_POLICY), else
/// <see cref="Allow"/>; one that is not registered is refused with "not registered", and nothing falls back to another
/// (a policy is a safety boundary).
/// </summary>
public static class WritePolicies
{
    /// <summary>Every write inside the workspace happens (the library's, the default).</summary>
    public const string Allow = "allow";

    /// <summary>Each write is asked about first; registered by an app (<see cref="AskingWritePolicy"/>), never by the library.</summary>
    public const string Ask = "ask";

    /// <summary>The variable that names the write policy when an option does not: "IDRAK_WRITE_POLICY".</summary>
    public const string EnvironmentVariable = "IDRAK_WRITE_POLICY";

    private static readonly SlotTable<string, IWritePolicy> Table = new(nameof(WritePolicies), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a write policy is a safety boundary: falling back to another when one fails would write a file the user was to be asked about");

    static WritePolicies() => Overrides.AsLibraryDefaults(() => Register(new AllowWritePolicy()));

    /// <summary>The policy named by <see cref="EnvironmentVariable"/>, or <see cref="Allow"/> when it is not set.</summary>
    public static string DefaultName => System.Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Allow;

    /// <summary>Registers <paramref name="policy"/> under its <see cref="IWritePolicy.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IWritePolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentException.ThrowIfNullOrWhiteSpace(policy.Name);
        Table.Register(policy.Name, policy, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's policy <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The policy registered as <paramref name="name"/>, or null.</summary>
    public static IWritePolicy? Find(string name) => Table.Find(name);

    /// <summary>The policy registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and who registers <see cref="Ask"/>.</exception>
    public static IWritePolicy Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(WritePolicies), name, Table.Keys,
            $"Write policy '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); "
            + $"add it with WritePolicies.Register; '{Allow}' comes with Idrak.Text, '{Ask}' is an app's (an AskingWritePolicy with its own question; "
            + "the CLI's idrak agent registers one).");

    /// <summary>The library's policy <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IWritePolicy? Default(string name) => Table.Default(name);

    /// <summary>Who registered the policy <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered policy as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}: {p.Summary}"));

    // The library's: every write inside the workspace happens (the tools have already refused one outside it).
    private sealed class AllowWritePolicy : IWritePolicy
    {
        public string Name => Allow;

        public string Summary => "every write inside the workspace happens, without asking";

        public ValueTask<string?> CheckAsync(FileWrite write, CancellationToken cancellationToken = default) => ValueTask.FromResult<string?>(null);
    }
}

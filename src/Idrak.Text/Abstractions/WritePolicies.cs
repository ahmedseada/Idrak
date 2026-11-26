// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
/// lets every write inside the workspace happen; an app's may ask its user (<see cref="AskingWritePolicy"/>,
/// <see cref="WritePolicies.Ask"/>), keep a log, or refuse some files. It is asked once per write, after the tool has
/// found what to write and before it writes; a write it refuses leaves the file as it was, and the model reads the reason.
/// Thread-safe: the tools of several agents may ask at once.
/// </summary>
public interface IWritePolicy
{
    /// <summary>Its name in <see cref="WritePolicies"/> ("allow", "ask").</summary>
    string Name { get; }

    /// <summary>Null to let <paramref name="write"/> happen; else why not, in words the model reads ("the user said no").</summary>
    /// <param name="write">The write, before it is made.</param>
    /// <param name="cancellationToken">Cancels the question (the tool call ends with it).</param>
    ValueTask<string?> CheckAsync(FileWrite write, CancellationToken cancellationToken = default);
}

/// <summary>
/// A write policy that asks (an app's user, a reviewer, a rule of its own) through <paramref name="ask"/>: true lets the
/// write happen, false refuses it with <see cref="Refusal"/>. Named <see cref="WritePolicies.Ask"/> unless another name is
/// given; the library makes none, since only an app has someone to ask.
/// </summary>
/// <param name="ask">The question: true to write, false to refuse.</param>
/// <param name="name">Its name in <see cref="WritePolicies"/> (<see cref="WritePolicies.Ask"/> by default).</param>
public sealed class AskingWritePolicy(Func<FileWrite, CancellationToken, ValueTask<bool>> ask, string name = WritePolicies.Ask) : IWritePolicy
{
    private readonly Func<FileWrite, CancellationToken, ValueTask<bool>> _ask = ask ?? throw new ArgumentNullException(nameof(ask));

    /// <inheritdoc />
    public string Name { get; } = string.IsNullOrWhiteSpace(name) ? throw new ArgumentException("A write policy needs a name.", nameof(name)) : name;

    /// <inheritdoc />


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
/// The names of the write policies of the coding tools' <c>write_file</c> and <c>edit_file</c>: <see cref="Allow"/>
/// (every write inside the workspace happens: the default) and <see cref="Ask"/> (an <see cref="AskingWritePolicy"/>
/// with its own question). The tools name theirs in their options' <c>WritePolicy</c>, else
/// <see cref="EnvironmentVariable"/> (IDRAK_WRITE_POLICY), else <see cref="Allow"/>; nothing falls back to another (a
/// policy is a safety boundary).
/// </summary>
public static class WritePolicies
{
    /// <summary>Every write inside the workspace happens (the library's, the default).</summary>
    public const string Allow = "allow";

    /// <summary>Each write is asked about first; an app's (<see cref="AskingWritePolicy"/>), never the library's.</summary>
    public const string Ask = "ask";

    /// <summary>The variable that names the write policy when an option does not: "IDRAK_WRITE_POLICY".</summary>
    public const string EnvironmentVariable = "IDRAK_WRITE_POLICY";

    /// <summary>The policy named by <see cref="EnvironmentVariable"/>, or <see cref="Allow"/> when it is not set.</summary>
    public static string DefaultName => System.Environment.GetEnvironmentVariable(EnvironmentVariable) is { } name && name.Trim().Length > 0 ? name.Trim() : Allow;

    // The library's: every write inside the workspace happens (the tools have already refused one outside it).
    internal sealed class AllowWritePolicy : IWritePolicy
    {
        public string Name => Allow;

        public ValueTask<string?> CheckAsync(FileWrite write, CancellationToken cancellationToken = default) => ValueTask.FromResult<string?>(null);
    }
}

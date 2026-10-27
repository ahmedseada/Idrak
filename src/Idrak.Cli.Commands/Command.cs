// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli;

/// <summary>One subcommand of <c>idrak</c> (or of a group, such as <c>idrak cache info</c>).</summary>
internal abstract class Command
{
    /// <summary>The words that select it after <c>idrak</c>, e.g. "doctor" or "cache info".</summary>
    public abstract string Name { get; }

    /// <summary>Other names that select it, e.g. "ls" for "list" (plans/done/idrak-cli.md, "Short forms").</summary>
    public virtual IReadOnlyCollection<string> Aliases => [];

    /// <summary>One line for <c>idrak help</c>.</summary>
    public abstract string Summary { get; }

    /// <summary>The command's own arguments and options, shown by <c>idrak help NAME</c> (the common options are added).</summary>
    public virtual string Usage => "";

    /// <summary>Options of this command that take a value (e.g. "--port").</summary>
    public virtual IReadOnlyCollection<string> ValueOptions => [];

    /// <summary>Flags of this command (e.g. "--int8"); any other option is a usage error.</summary>
    public virtual IReadOnlyCollection<string> Flags => [];

    /// <summary>
    /// One-letter short forms of this command's options, short to long (e.g. "-p" to "--port"); the common ones
    /// (<see cref="CommandContext.CommonShortForms"/>) apply to every command and may not be given another meaning.
    /// </summary>
    public virtual IReadOnlyDictionary<string, string> ShortForms => EmptyShortForms;

    private static readonly Dictionary<string, string> EmptyShortForms = [];

    /// <summary>
    /// Values of the common <c>--format</c> this command reads itself beyond text, json, csv and md (transcribe's srt and
    /// vtt): with one of them the context's <see cref="CommandContext.Format"/> is text and the command reads the option.
    /// </summary>
    public virtual IReadOnlyCollection<string> OwnFormats => [];

    /// <summary>
    /// What it always needs from the process it runs in (<see cref="CommandNeeds"/>: another program, a network listener, the
    /// console's keyboard); a host that does not allow it refuses the command with that reason before it runs. None by default.
    /// </summary>
    public virtual CommandNeeds Needs => CommandNeeds.None;

    /// <summary>What it needs with the options of <paramref name="context"/> (<see cref="Needs"/> unless an option changes it).</summary>
    public virtual CommandNeeds NeedsFor(CommandContext context) => Needs;

    /// <summary>Called before the <c>--plugin</c> assemblies load (e.g. to note what was registered before them).</summary>
    public virtual void BeforePlugins(CommandContext context)
    {
    }

    /// <summary>Runs the command; returns the exit code (<see cref="ExitCodes"/>).</summary>
    public abstract int Run(CommandContext context);
}

/// <summary>The tool's exit codes.</summary>
internal static class ExitCodes
{
    public const int Ok = 0;
    public const int Failed = 1;
    public const int Usage = 2;

    /// <summary>The command needs what its host does not allow (<see cref="CommandHost.Allows"/>).</summary>
    public const int Unavailable = 3;

    /// <summary>The host stopped the command (an app's Stop; the cancellation given to <see cref="CommandLine.Run(IReadOnlyList{string}, TextWriter, TextWriter, CommandHost, CancellationToken)"/>).</summary>
    public const int Stopped = 130;
}

/// <summary>Thrown for a usage error (unknown option, missing argument): the tool prints it and exits with 2.</summary>
internal sealed class UsageException(string message) : Exception(message);

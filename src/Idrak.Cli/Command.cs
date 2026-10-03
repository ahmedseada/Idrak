// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli;

/// <summary>One subcommand of <c>idrak</c> (or of a group, such as <c>idrak cache info</c>).</summary>
internal abstract class Command
{
    /// <summary>The words that select it after <c>idrak</c>, e.g. "doctor" or "cache info".</summary>
    public abstract string Name { get; }

    /// <summary>One line for <c>idrak help</c>.</summary>
    public abstract string Summary { get; }

    /// <summary>The command's own arguments and options, shown by <c>idrak help NAME</c> (the common options are added).</summary>
    public virtual string Usage => "";

    /// <summary>Options of this command that take a value (e.g. "--port").</summary>
    public virtual IReadOnlyCollection<string> ValueOptions => [];

    /// <summary>Flags of this command (e.g. "--int8"); any other option is a usage error.</summary>
    public virtual IReadOnlyCollection<string> Flags => [];

    /// <summary>Runs the command; returns the exit code (<see cref="ExitCodes"/>).</summary>
    public abstract int Run(CommandContext context);
}

/// <summary>The tool's exit codes.</summary>
internal static class ExitCodes
{
    public const int Ok = 0;
    public const int Failed = 1;
    public const int Usage = 2;
}

/// <summary>Thrown for a usage error (unknown option, missing argument): the tool prints it and exits with 2.</summary>
internal sealed class UsageException(string message) : Exception(message);

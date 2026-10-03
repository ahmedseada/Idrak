// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

/// <summary>
/// Where commands read typed or piped input (chat's lines, <c>run</c>'s piped prompt, agent questions): the console by
/// default; tests set <see cref="Override"/> to a reader of their own, so a conversation runs in-process.
/// </summary>
internal static class StandardInput
{
    /// <summary>A reader used instead of the console (tests), or null.</summary>
    public static TextReader? Override { get; set; }

    /// <summary>The input.</summary>
    public static TextReader Reader => Override ?? Console.In;

    /// <summary>Whether the input is piped or a file rather than a person typing (always true with <see cref="Override"/>).</summary>
    public static bool IsRedirected => Override is not null || Console.IsInputRedirected;

    /// <summary>Runs <paramref name="action"/> with <paramref name="input"/> as the input (tests).</summary>
    public static T With<T>(TextReader input, Func<T> action)
    {
        var previous = Override;
        Override = input;
        try
        {
            return action();
        }
        finally
        {
            Override = previous;
        }
    }
}

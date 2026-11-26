// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Commands;

/// <summary>
/// The audio commands (plan 31, P4): sound run through an audio pipeline (<c>audio process</c>).
/// </summary>
internal static class AudioCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new AudioProcessCommand(),
    ];
}

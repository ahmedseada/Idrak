// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Commands;

/// <summary>
/// The audio commands (plan 31, P4): sound run through an audio pipeline (<c>audio process</c>) and the processors a
/// pipeline can name (<c>audio processors</c>), the registered ones (<c>Idrak.Audio.Abstractions.AudioProcessors</c>).
/// </summary>
internal static class AudioCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new AudioProcessCommand(),
        new AudioProcessorsCommand(),
    ];
}

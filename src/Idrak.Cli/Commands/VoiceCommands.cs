// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Voices;

namespace Idrak.Cli.Commands;

/// <summary>
/// The voice commands (plan 21, S9's command line): a person's voice installed by name from a few clips, with their
/// consent, and listed, shown, removed, exported and imported; <c>idrak speak --voice NAME</c> speaks with it.
/// </summary>
internal static class VoiceCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new VoiceAddCommand(),
        new VoiceListCommand(),
        new VoiceShowCommand(),
        new VoiceRemoveCommand(),
        new VoiceExportCommand(),
        new VoiceImportCommand(),
    ];
}

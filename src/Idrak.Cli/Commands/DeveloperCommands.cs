// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Developer;

namespace Idrak.Cli.Commands;

/// <summary>The new, test, onnx, kernels dump, trace, demo and shell commands (plans/idrak-cli.md, "Developers").</summary>
internal static class DeveloperCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new NewCommand(),
        new TestCommand(),
        new OnnxImportCommand(),
        new OnnxExportCommand(),
        new OnnxCheckCommand(),
        new KernelsDumpCommand(),
        new TraceCommand(),
        new DemoCommand(),
        new ShellCommand(),
    ];
}

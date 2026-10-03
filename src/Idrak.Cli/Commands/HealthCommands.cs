// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Health;

namespace Idrak.Cli.Commands;

/// <summary>Setup and health: doctor, devices, version, report, env, init, cache, config, plugins, formats, completion, update, login, setup (plans/idrak-cli.md).</summary>
internal static class HealthCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new DoctorCommand(), new DevicesCommand(), new VersionCommand(), new ReportCommand(), new EnvCommand(), new InitCommand(),
        new CacheInfoCommand(), new CacheClearCommand(), new ConfigGetCommand(), new ConfigSetCommand(), new ConfigUnsetCommand(),
        new ConfigListCommand(), new PluginsListCommand(), new FormatsCommand(), new CompletionCommand(), new UpdateCommand(),
        new LoginCommand(), new LogoutCommand(), new SetupAndroidCommand(),
    ];
}

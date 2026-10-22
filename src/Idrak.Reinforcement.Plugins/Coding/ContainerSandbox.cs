// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Text.Generation;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// Settings of a <see cref="ContainerSandbox"/>. The program, the image and the network read their variables when the
/// options are made (a value given in code wins): <see cref="ProgramVariable"/>, <see cref="ImageVariable"/>,
/// <see cref="NetworkVariable"/>. No image is assumed: without one the sandbox refuses.
/// </summary>
public sealed record ContainerSandboxOptions
{
    /// <summary>The variable of <see cref="Program"/>: "IDRAK_SANDBOX_PROGRAM".</summary>
    public const string ProgramVariable = "IDRAK_SANDBOX_PROGRAM";

    /// <summary>The variable of <see cref="Image"/>: "IDRAK_SANDBOX_IMAGE".</summary>
    public const string ImageVariable = "IDRAK_SANDBOX_IMAGE";

    /// <summary>The variable of <see cref="Network"/>: "IDRAK_SANDBOX_NETWORK".</summary>
    public const string NetworkVariable = "IDRAK_SANDBOX_NETWORK";

    /// <summary>The container program (a path, or a name looked for on the PATH); null: the first of <see cref="Programs"/> found on the PATH.</summary>
    public string? Program { get; init; } = Setting(ProgramVariable);

    /// <summary>The programs looked for on the PATH, in order, when <see cref="Program"/> is not set: docker, then podman.</summary>
    public IReadOnlyList<string> Programs { get; init; } = ["docker", "podman"];

    /// <summary>The image every command runs in (it holds the tools the commands start); null: none, so the sandbox refuses.</summary>
    public string? Image { get; init; } = Setting(ImageVariable);

    /// <summary>
    /// The containers' network when a command may reach one (<see cref="CommandLimits.Network"/>): null, "off", "none" or
    /// "false" (the default): none; "on" or "true": the program's default network; any other value: that network's name.
    /// </summary>
    public string? Network { get; init; } = Setting(NetworkVariable);

    /// <summary>Where the workspace (<see cref="CommandLimits.Root"/>) is mounted in the container ("/workspace").</summary>
    public string MountPoint { get; init; } = "/workspace";

    /// <summary>The user the commands run as in the container (the program's --user, "uid:gid"); null: the image's.</summary>
    public string? User { get; init; }

    /// <summary>More arguments given to the program's run, before the image (a security option, a cache mount); null for none.</summary>
    public IReadOnlyList<string>? ExtraArguments { get; init; }

    /// <summary>The longest the removal of a stopped container may take.</summary>
    public TimeSpan RemoveTimeout { get; init; } = TimeSpan.FromSeconds(30);

    internal bool NetworkOn => Network?.Trim().ToLowerInvariant() is { Length: > 0 } value && value is not ("off" or "none" or "false" or "0");

    private static string? Setting(string name) => Environment.GetEnvironmentVariable(name) is { } value && value.Trim().Length > 0 ? value.Trim() : null;
}

/// <summary>
/// The <c>container</c> command sandbox (<see cref="CommandSandboxes.Container"/>, registered by <see cref="CodingPlugin"/>):
/// each command runs in a fresh container of <see cref="ContainerSandboxOptions.Image"/> started by Docker or Podman
/// (<c>run --rm</c>), the workspace mounted at <see cref="ContainerSandboxOptions.MountPoint"/>, the network off unless
/// allowed, and the memory, CPU and process limits of <see cref="CommandLimits"/> given to the program, which enforces them
/// for the whole process tree. The program is started through the library's plain-process sandbox (its time limit and
/// output cap); a container still running when the time is up or the call is cancelled is removed. It refuses loudly when
/// no program is found or no image is set. Running it needs Docker or Podman on the machine.
/// </summary>
/// <param name="options">The program, the image, the network and the mount; null for the defaults (and the variables').</param>
public sealed class ContainerSandbox(ContainerSandboxOptions? options = null) : ICommandSandbox
{
    /// <summary>The settings.</summary>
    public ContainerSandboxOptions Options { get; } = options ?? new ContainerSandboxOptions();

    /// <inheritdoc />
    public string Name => CommandSandboxes.Container;

    /// <inheritdoc />
    public string Summary => "a Docker or Podman container per command: the workspace mounted, the network off unless allowed, memory, CPU and process limits (IDRAK_SANDBOX_IMAGE)";

    /// <summary>The container program's path: <see cref="ContainerSandboxOptions.Program"/> (a path, or a name on the PATH), else the first of <see cref="ContainerSandboxOptions.Programs"/> on the PATH; null when none is found.</summary>
    public string? FindProgram()
    {
        if (Options.Program is { } program)
        {
            return File.Exists(program) ? Path.GetFullPath(program) : Path.IsPathRooted(program) ? null : ProcessCommandSandbox.FindProgram(program);
        }

        foreach (var name in Options.Programs)
        {
            if (ProcessCommandSandbox.FindProgram(name) is { } found)
            {
                return found;
            }
        }

        return null;
    }

    /// <inheritdoc />
    public string? Check(CommandLimits limits)
    {
        ArgumentNullException.ThrowIfNull(limits);
        if (FindProgram() is null)
        {
            return Options.Program is { } program
                ? $"The {Name} sandbox's program '{program}' ({ContainerSandboxOptions.ProgramVariable}) is not found; give the path of docker or podman."
                : $"The {Name} sandbox found no container program: none of {string.Join(", ", Options.Programs)} is on the PATH; install Docker or Podman, "
                  + $"or set {ContainerSandboxOptions.ProgramVariable} to the program's path.";
        }

        return Options.Image is null
            ? $"The {Name} sandbox has no image: set {ContainerSandboxOptions.ImageVariable} (or ContainerSandboxOptions.Image) to an image that holds the programs the commands run; none is assumed."
            : null;
    }

    /// <summary>
    /// The command line that runs <paramref name="command"/> in a container named <paramref name="name"/>:
    /// <paramref name="program"/> <c>run --rm --name</c> …, the network, the workspace mount and working folder, the limits,
    /// the user, the variables, <see cref="ContainerSandboxOptions.ExtraArguments"/>, the image, then the command.
    /// </summary>
    /// <exception cref="ArgumentException">The folder is not inside <see cref="CommandLimits.Root"/>.</exception>
    /// <exception cref="NotSupportedException">No image is set.</exception>
    public IReadOnlyList<string> CommandLine(string program, IReadOnlyList<string> command, string directory, CommandLimits limits, string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(program);
        ArgumentNullException.ThrowIfNull(command);
        ArgumentNullException.ThrowIfNull(directory);
        ArgumentNullException.ThrowIfNull(limits);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        string image = Options.Image ?? throw new NotSupportedException(Check(limits) ?? $"The {Name} sandbox has no image.");
        string root = Path.GetFullPath(limits.Root ?? directory), folder = Path.GetFullPath(directory);
        string relative = Path.GetRelativePath(root, folder);
        if (relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || Path.IsPathRooted(relative))
        {
            throw new ArgumentException($"The folder '{directory}' is not inside the sandbox's root '{root}'.", nameof(directory));
        }

        string mount = Options.MountPoint.TrimEnd('/');
        string workdir = relative == "." ? mount : $"{mount}/{relative.Replace('\\', '/')}";
        var invariant = CultureInfo.InvariantCulture;
        var line = new List<string> { program, "run", "--rm", "--name", name };
        if (!limits.Network || !Options.NetworkOn)
        {
            line.AddRange(["--network", "none"]);
        }
        else if (Options.Network!.Trim().ToLowerInvariant() is not ("on" or "true" or "1"))
        {
            line.AddRange(["--network", Options.Network.Trim()]);
        }

        line.AddRange(["--mount", $"type=bind,source={root},target={mount}", "--workdir", workdir]);
        if (limits.MaxMemoryBytes > 0)
        {
            string bytes = limits.MaxMemoryBytes.ToString(invariant) + "b";
            line.AddRange(["--memory", bytes, "--memory-swap", bytes]);
        }

        if (limits.MaxCpus > 0)
        {
            line.AddRange(["--cpus", limits.MaxCpus.ToString("0.###", invariant)]);
        }

        if (limits.MaxProcesses > 0)
        {
            line.AddRange(["--pids-limit", limits.MaxProcesses.ToString(invariant)]);
        }

        if (Options.User is { } user)
        {
            line.AddRange(["--user", user]);
        }

        foreach (var (key, value) in (limits.Environment ?? new Dictionary<string, string>()).OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            line.AddRange(["--env", $"{key}={value}"]);
        }

        line.AddRange(Options.ExtraArguments ?? []);
        line.Add(image);
        line.AddRange(command);
        return line;
    }

    /// <inheritdoc />
    public async ValueTask<CommandResult> RunAsync(IReadOnlyList<string> command, string directory, CommandLimits limits, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(command);
        ArgumentNullException.ThrowIfNull(directory);
        ArgumentNullException.ThrowIfNull(limits);
        if (Check(limits) is { } refusal)
        {
            throw new NotSupportedException(refusal);
        }

        static CommandResult Refuse(string why) => new(null, false, why, TimeSpan.Zero, true);
        if (command.Count == 0 || command[0].Length == 0)
        {
            return Refuse("Error: empty command.");
        }

        string programName = Path.GetFileNameWithoutExtension(command[0]);
        if (limits.Programs is { } allowed && (!allowed.Contains(programName, StringComparer.OrdinalIgnoreCase) || command[0].Contains('/') || command[0].Contains('\\')))
        {
            return Refuse($"Error: '{command[0]}' is not an allowed program. Allowed: {string.Join(", ", allowed)}.");
        }

        string program = FindProgram()!;
        string name = "idrak-sandbox-" + Guid.NewGuid().ToString("N");
        IReadOnlyList<string> line;
        try
        {
            line = CommandLine(program, command, directory, limits, name);
        }
        catch (ArgumentException e)
        {
            return Refuse($"Error: {e.Message}");
        }

        // The program itself runs as a plain process under the time limit and the output cap; the container holds the rest.
        var process = CommandSandboxes.Default(CommandSandboxes.Process) ?? new ProcessCommandSandbox();
        var outer = new CommandLimits { Time = limits.Time, MaxOutputCharacters = limits.MaxOutputCharacters };
        try
        {
            var result = await process.RunAsync(line, directory, outer, cancellationToken).ConfigureAwait(false);
            if (result.TimedOut)
            {
                await RemoveAsync(process, program, name).ConfigureAwait(false);
            }

            return result;
        }
        catch (OperationCanceledException)
        {
            await RemoveAsync(process, program, name).ConfigureAwait(false);
            throw;
        }
    }

    // Removes a container that may still run (best effort: its run was stopped).
    private async Task RemoveAsync(ICommandSandbox process, string program, string name)
    {
        try
        {
            await process.RunAsync([program, "rm", "--force", name], Path.GetTempPath(), new CommandLimits { Time = Options.RemoveTimeout }).ConfigureAwait(false);
        }
        catch (Exception e) when (e is IOException or InvalidOperationException or OperationCanceledException)
        {
        }
    }
}

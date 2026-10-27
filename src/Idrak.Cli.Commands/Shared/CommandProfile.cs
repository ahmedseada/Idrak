// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.RegularExpressions;
using Idrak.Cli.Commands.Health;

namespace Idrak.Cli.Shared;

/// <summary>
/// <c>--profile</c> (or <c>IDRAK_PROFILE=1</c>), common to every command: the command runs as it would without it while the
/// library's watch (<see cref="Profiler.Watch"/>) measures it from outside with the probes that do not change it (time on
/// the processor, the managed heap, device memory, kernel choices measured, host fallbacks, the stages of reading answers,
/// and any a package registered). When the command ends, normally, by an error or by Ctrl+C, one report is written with
/// the <c>IDRAK_PROFILE_FORMAT</c> writer to the file <c>IDRAK_PROFILE_OUT</c> names (appended), else to the error stream,
/// so the main output (and <c>--json</c>) stays as it is. <c>IDRAK_PROFILE_PROBES</c> chooses the probes.
/// </summary>
internal sealed partial class CommandProfile
{
    /// <summary>The variable that turns <c>--profile</c> on for every command: 1, true, on or yes (0, false, off or no: off).</summary>
    internal const string Variable = "IDRAK_PROFILE";

    private readonly Lock _lock = new();
    private readonly ProfileSession _session;
    private readonly ProfileOptions _options;
    private readonly TextWriter _error;
    private readonly long _start = Stopwatch.GetTimestamp();
    private bool _written;

    private CommandProfile(ProfileSession session, ProfileOptions options, TextWriter error)
    {
        (_session, _options, _error) = (session, options, error);
        Idrak.Cli.Shared.Terminal.AddCancelKey(OnCancel);
    }

    /// <summary>Whether the command is to be profiled: <c>--profile</c>, or <see cref="Variable"/> set to on.</summary>
    /// <exception cref="UsageException">The variable holds something else than on or off.</exception>
    internal static bool Requested(CommandContext context)
    {
        if (context.Flag("--profile"))
        {
            return true;
        }

        string? value = Environment.GetEnvironmentVariable(Variable)?.Trim();
        return value?.ToLowerInvariant() switch
        {
            null or "" or "0" or "false" or "off" or "no" => false,
            "1" or "true" or "on" or "yes" => true,
            _ => throw new UsageException($"{Variable}={value}: give 1 (profile every command, as --profile does) or 0; the config profile's variable is IDRAK_CONFIG_PROFILE."),
        };
    }

    /// <summary>
    /// Starts watching the command <paramref name="args"/> name on its device (<c>--device</c>, the config, else the
    /// default; the CPU, with a note, when that cannot start) when <see cref="Requested"/>; null otherwise.
    /// </summary>
    /// <exception cref="UsageException">An <c>IDRAK_PROFILE_*</c> value that does not parse, or a probe or writer that is not registered.</exception>
    internal static CommandProfile? Start(CommandContext context, IReadOnlyList<string> args)
    {
        if (!Requested(context))
        {
            return null;
        }

        ProfileOptions options;
        try
        {
            options = ProfileOptions.FromEnvironment();
            _ = ProfileWriters.Get(options.Format);                             // a writer that is not there fails now, not after the command
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            throw new UsageException($"--profile: {e.Message}");
        }

        Device device;
        string? deviceNote = null;
        try
        {
            device = context.Device;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            device = Device.Cpu;
            deviceNote = $"the command's device did not start ({e.Message.Split('\n')[0]}): profiled on the CPU";
        }

        ProfileSession session;
        try
        {
            session = Profiler.Watch(Shown(args), device, options);
        }
        catch (NotSupportedException e)
        {
            throw new UsageException($"--profile: {e.Message}");
        }

        if (deviceNote is not null)
        {
            session.Note(deviceNote);
        }

        return new CommandProfile(session, options, context.ErrorOutput);
    }

    /// <summary>Ends the watch and writes the report, with how the command ended: its exit code and, after an error, its message.</summary>
    internal void End(int exit, string? failure) =>
        Finish(failure is null ? $"ended with exit code {exit}" : $"failed with exit code {exit}: {failure.Split('\n')[0]}");

    // The first Ctrl+C: a command that stops cleanly goes on to stop, one that does not is ended by the runtime after the
    // handlers, so the report is written now, either way, with what was measured until then.
    private void OnCancel(object? sender, ConsoleCancelEventArgs e) =>
        Finish("interrupted by Ctrl+C: the report holds what was measured until then (a command that stops cleanly goes on after it)");

    private void Finish(string how)
    {
        lock (_lock)
        {
            if (_written)
            {
                return;
            }

            _written = true;
            Idrak.Cli.Shared.Terminal.RemoveCancelKey(OnCancel);
            var elapsed = Stopwatch.GetElapsedTime(_start);
            string wall = elapsed.TotalMinutes < 1 ? elapsed.TotalSeconds.ToString("0.00 s", System.Globalization.CultureInfo.InvariantCulture) : Units.Duration(elapsed);
            _session.Note($"wall time {wall}; {how}");
            var report = _session.End();

            try
            {
                if (_options.Output is { } path)
                {
                    string full = Path.GetFullPath(path);
                    Directory.CreateDirectory(Path.GetDirectoryName(full)!);
                    using var file = new StreamWriter(full, append: true);
                    report.Write(file);
                    _error.WriteLine($"idrak: profile written to {full}");
                }
                else
                {
                    _error.WriteLine();
                    report.Write(_error);
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                _error.WriteLine($"idrak: the profile could not be written to {_options.Output}: {e.Message}");
            }
        }
    }

    /// <summary>
    /// The command line as the report shows it: <c>idrak</c> and the arguments, quoted where they hold spaces, with every
    /// secret hidden: the value of an option with a word token, key, password or secret in its name (<c>--hf-token X</c>,
    /// <c>--api-key=X</c>; not <c>--tokenizer</c>), the value after a config key for a secret or a secret environment
    /// variable's name (<c>config set tokens.github X</c>, <c>env set HF_TOKEN X</c>, <c>HF_TOKEN=X</c>), and a password
    /// in an address.
    /// </summary>
    internal static string Shown(IReadOnlyList<string> args)
    {
        const string Hidden = "***";
        var shown = new List<string>(args.Count + 1) { "idrak" };
        bool hideNext = false;
        foreach (string arg in args)
        {
            string text = arg;
            if (hideNext)
            {
                text = Hidden;
                hideNext = false;
            }
            else if (arg.StartsWith('-') && arg.Length > 1)
            {
                int equals = arg.IndexOf('=');
                string name = equals > 0 ? arg[..equals] : arg;
                if (name.TrimStart('-').Split('-').Any(w => w is "token" or "tokens" or "key" or "keys" or "password" or "secret" or "secrets" or "apikey"))
                {
                    if (equals > 0)
                    {
                        text = name + "=" + Hidden;
                    }
                    else
                    {
                        hideNext = true;
                    }
                }
            }
            else if (arg.IndexOf('=') is > 0 and int at && Secret(arg[..at]))
            {
                text = arg[..at] + "=" + Hidden;                                  // NAME=VALUE
            }
            else if (Secret(arg))
            {
                hideNext = true;                                                  // a secret's name: its value follows
            }

            text = Address().Replace(text, "$1" + Hidden + "@");
            shown.Add(text.Length == 0 || text.Any(char.IsWhiteSpace) ? "\"" + text.Replace("\"", "\\\"", StringComparison.Ordinal) + "\"" : text);
        }

        return string.Join(' ', shown);

        // A secret's name as an argument: a secret environment variable, or a dotted config key for one (tokens.github).
        static bool Secret(string name) => EnvironmentVariables.All.Any(v => v.Secret && v.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            || (name.Contains('.') && !name.Contains('/') && !name.Contains('\\') && ConfigCommand.IsSecret(name));
    }

    // The password of an address (scheme://user:password@host).
    [GeneratedRegex(@"(\w+://[^/\s:@]+:)[^/\s@]+@")]
    private static partial Regex Address();
}

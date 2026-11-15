// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Developer.Stacks;

/// <summary>
/// <c>idrak trace --stacks [options] -- COMMAND ...</c>: runs COMMAND in a child process with the .NET runtime's sampling on,
/// then reports where every thread's time went (<see cref="StackReport"/>); <c>--read FILE</c> reports a trace made before
/// (any window, focus or comparison, without running again).
/// </summary>
internal static class TraceStacks
{
    /// <summary>The value options of the stacks report.</summary>
    public static readonly string[] ValueOptions = ["--read", "--focus", "--top", "--from", "--to", "--slices", "--compare"];

    /// <summary>The flags of the stacks report.</summary>
    public static readonly string[] Flags = ["--stacks", "--waits", "--all-frames", "--by-thread", "--timeline"];

    /// <summary>Whether the command line asks for the stacks report.</summary>
    public static bool Asked(CommandContext context) => context.Flag("--stacks") || context.Option("--read") is not null;

    /// <summary>A stacks option given without --stacks or --read, or null.</summary>
    public static string? Stray(CommandContext context) =>
        ValueOptions.FirstOrDefault(o => context.Option(o) is not null) ?? Flags.Skip(1).FirstOrDefault(context.Flag);

    public static int Run(CommandContext context, Func<CommandContext, List<string>> forwarded)
    {
        if (context.Option("--levels") is not null || context.Flag("--sync"))
        {
            throw new UsageException("--levels and --sync are the telemetry trace's; --stacks samples the stacks instead (give one or the other).");
        }

        int top = context.IntOption("--top", 15);
        int slices = context.IntOption("--slices", 20);
        if (top < 1 || slices < 1)
        {
            throw new UsageException("--top and --slices need a count of 1 or more.");
        }

        var options = new StackReportOptions(top, context.Option("--focus"), context.Flag("--waits"), context.Flag("--all-frames"), context.Flag("--by-thread"),
            context.Flag("--timeline") ? slices : 0);
        double? from = Seconds(context.Option("--from"), "--from"), to = Seconds(context.Option("--to"), "--to");
        if (from is { } f && to is { } t && t <= f)
        {
            throw new UsageException("--to must come after --from.");
        }

        string? read = context.Option("--read");
        int exit = 0;
        string path;
        string? command = null, result = null;
        if (read is null)
        {
            if (context.Positional.Count == 0)
            {
                throw new UsageException("Missing the COMMAND to trace (e.g. idrak trace --stacks -- bench Qwen/Qwen3-0.6B -d cpu), or --read FILE for a trace made before.");
            }

            if (CommandLine.Find(CommandTable.All, context.Positional, out _) is not { } inner)
            {
                throw new UsageException($"Unknown command '{context.Positional[0]}'. Run 'idrak help' for the commands.");
            }

            if (inner is TraceCommand)
            {
                throw new UsageException("idrak trace cannot trace itself.");
            }

            command = string.Join(' ', context.Positional);
            path = Path.GetFullPath(context.Option("--out")
                ?? Path.Combine("test-output", "trace", $"{string.Join('-', inner.Name.Split(' '))}-{DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)}.nettrace"));
            (context.Json ? context.ErrorOutput : context.Output).WriteLine($"trace: sampling the stacks of 'idrak {command}' into {path}");
            var output = context.Json ? new StringWriter() : context.Output;
            exit = StackRecorder.Record(StackRecorder.Launcher(forwarded(context)), path, output, context.ErrorOutput);
            result = context.Json ? output.ToString() : null;
            context.Write($"trace: {inner.Name} exited with {exit}");
        }
        else
        {
            if (context.Positional.Count > 0)
            {
                throw new UsageException("--read reports a trace made before: give no COMMAND with it.");
            }

            path = Path.GetFullPath(read);
        }

        var profile = StackProfile.Of(NetTrace.Read(path), from, to);
        var old = context.Option("--compare") is { } compare ? StackProfile.Of(NetTrace.Read(Path.GetFullPath(compare)), from, to) : null;
        var json = context.Json ? new JsonObject() : null;
        StackReport.Write(context, profile, options, old, json);
        string speedscope = Path.ChangeExtension(path, ".speedscope.json");
        Speedscope.Write(profile, speedscope);
        context.Write($"files: the trace {path}; its stacks for a viewer {speedscope} (https://www.speedscope.app)");
        if (json is not null)
        {
            JsonNode? inner = null;
            if (!string.IsNullOrEmpty(result))
            {
                try
                {
                    inner = JsonNode.Parse(result);
                }
                catch (System.Text.Json.JsonException)
                {
                    inner = result;
                }
            }

            var document = new JsonObject { ["command"] = command, ["exitCode"] = exit, ["trace"] = path, ["speedscope"] = speedscope, ["stacks"] = json, ["result"] = inner };
            context.WriteJson(document);
        }

        return exit;
    }

    // Seconds since the run's start ("1.5" or "1.5s").
    private static double? Seconds(string? text, string option)
    {
        if (text is null)
        {
            return null;
        }

        string value = text.EndsWith('s') ? text[..^1] : text;
        return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds) && seconds >= 0
            ? seconds
            : throw new UsageException($"{option} takes seconds since the run's start (e.g. 1.5), not '{text}'.");
    }
}

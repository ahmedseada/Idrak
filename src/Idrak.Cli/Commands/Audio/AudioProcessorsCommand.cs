// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio.Abstractions;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak audio processors [NAME...]</c> (plan 31, P4): every processor an audio pipeline can name
/// (<see cref="AudioProcessors"/>, the library's and the plug-ins'), whether it runs on live sound or needs the whole
/// recording (with its live alternative), its summary, and each of its inputs (<see cref="IAudioProcessor.Parameters"/>):
/// key, kind, range and unit, default and help.
/// </summary>
internal sealed class AudioProcessorsCommand : Command
{
    public override string Name => "audio processors";

    public override string Summary => "The audio processors a pipeline can name: live or whole recording, and their options";

    public override string Usage => """
        [NAME...]

        Lists every registered audio processor (the library's filters, dynamics and cleaning stages, and those -P
        plug-ins register) in the order registered: its name and label, whether it runs on live sound as it comes or
        needs the whole recording (and then the live processor to use instead), what it does, and each input: the value
        written after '=' (shown as =VALUE) and the options written KEY=VALUE after a space, with the kind, the range and
        unit, the default and what it changes. -j prints the same as JSON.

        A pipeline names processors in order, comma-separated: "highpass=80, dehum=50, denoise=1.5 quantile=0.2,
        normalize=-20 peak=0.9". After a comma a registered name starts a new step, so an option goes after a space.
        idrak audio process, idrak transcribe --audio-clean, idrak speak --audio-clean and --clean on pretrain, voice
        train and data audio take one.

        Arguments:
          NAME                   only these processors (default: all)

        Examples:
          idrak audio processors
          idrak audio processors denoise agc
          idrak audio processors -j
        """;

    public override int Run(CommandContext context)
    {
        IReadOnlyList<string> names = context.Positional.Count > 0 ? [.. context.Positional] : [.. AudioProcessors.Names];
        var processors = new List<IAudioProcessor>();
        foreach (string name in names)
        {
            processors.Add(AudioProcessors.Find(name)
                ?? throw new UsageException($"Unknown audio processor '{name}' (registered: {string.Join(", ", AudioProcessors.Names)})."));
        }

        var rows = new JsonArray();
        foreach (var processor in processors)
        {
            rows.Add((JsonNode)Json(processor));
            if (context.Json)
            {
                continue;
            }

            string how = processor.Live ? "live" : processor.LiveAlternative is { } alternative ? $"whole recording; live: {alternative}" : "whole recording";
            context.Write($"{processor.Name} ({processor.Label}): {how}");
            context.Write($"  {processor.Summary}");
            var inputs = processor.Parameters.Select(p => (IReadOnlyList<string>)
                [p.Key.Length == 0 ? "=VALUE" : p.Key, Kind(p), Range(p), p.Default is { } d ? $"default {d}" : p.Optional ? "optional" : "required", p.Help ?? ""]).ToList();
            if (inputs.Count == 0)
            {
                context.Write("  takes no value and no options");
            }
            else
            {
                int[] widths = [.. Enumerable.Range(0, 4).Select(c => inputs.Max(r => r[c].Length))];
                foreach (var input in inputs)
                {
                    context.Write(("    " + string.Join("  ", input.Take(4).Select((cell, c) => cell.PadRight(widths[c]))) + "  " + input[4]).TrimEnd());
                }
            }

            context.Write("");
        }

        if (!context.Json)
        {
            context.Write(string.Create(CultureInfo.InvariantCulture,
                $"{processors.Count} processor(s); a pipeline: \"highpass=80, denoise=1.5, normalize=-20 peak=0.9\" (idrak audio process IN -o OUT --pipeline P)"));
        }

        context.WriteJson(new JsonObject { ["processors"] = rows });
        return ExitCodes.Ok;
    }

    /// <summary>A processor as JSON: name, label, summary, live, its live alternative, who registered it, its parameters.</summary>
    internal static JsonObject Json(IAudioProcessor processor) => new()
    {
        ["name"] = processor.Name,
        ["label"] = processor.Label,
        ["summary"] = processor.Summary,
        ["live"] = processor.Live,
        ["live_alternative"] = processor.LiveAlternative,
        ["origin"] = AudioProcessors.Origin(processor.Name),
        ["parameters"] = new JsonArray([.. processor.Parameters.Select(p => (JsonNode)new JsonObject
        {
            ["key"] = p.Key,
            ["label"] = p.Label,
            ["kind"] = p.Input.ToString().ToLowerInvariant(),
            ["min"] = p.Min,
            ["min_exclusive"] = p.MinExclusive,
            ["max"] = p.Max,
            ["step"] = p.Step,
            ["unit"] = p.Unit,
            ["range"] = p.Range,
            ["choices"] = new JsonArray([.. p.Choices.Select(c => (JsonNode)c)]),
            ["default"] = p.Default,
            ["optional"] = p.Optional,
            ["help"] = p.Help,
        })]),
    };

    // The kind as people read it: number, whole number, one of the choices, true|false.
    private static string Kind(AudioProcessorParameter p) => p.Input switch
    {
        AudioProcessorInput.Integer => "whole number",
        AudioProcessorInput.Choice => string.Join("|", p.Choices),
        AudioProcessorInput.Switch => "true|false",
        _ => "number",
    };

    private static string Range(AudioProcessorParameter p) =>
        p.Range.Length == 0 ? p.Unit ?? "" : p.Unit is { Length: > 0 } unit ? $"{p.Range} {unit}" : p.Range;
}

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak audio process IN -o OUT --pipeline P</c> (plan 31, P4): a sound file decoded by the registered codecs, run
/// through an audio pipeline (<see cref="AudioPipeline"/>) and written by the encoder of the output's extension, to hear
/// what a pipeline does; it prints the steps that ran, the sound's length and the real-time factor. With
/// <c>--block-seconds</c> the sound goes through the pipeline's stage block by block, as live sound does.
/// </summary>
internal sealed class AudioProcessCommand : Command
{
    public override string Name => "audio process";

    public override string Summary => "Run a sound file through an audio pipeline (filters, dynamics, noise) and write the result";

    public override string Usage => """
        IN -o OUT --pipeline P [options]

        Decodes IN with the registered codecs (WAV, FLAC, MP3; another format with its plug-in, -P), runs it through the
        audio pipeline P and writes OUT with the encoder of its extension (.wav, .mp3, or a plug-in's), to hear what the
        pipeline does. Prints each step that ran, the sound's length before and after, the time taken and the real-time
        factor; -v runs it again warm and prints both (the first call compiles the code). P names processors in order,
        comma-separated, each with its value and KEY=VALUE options after a space ("highpass=80, dehum=50, denoise=1.5
        quantile=0.2, normalize=-20 peak=0.9"); idrak audio processors lists them with their options.

        Arguments:
          IN                     the sound file, or - for standard input

        Options:
          -o, --out FILE         the file to write (its extension chooses the encoder)
              --pipeline P       the pipeline: its text, a JSON array of steps (["highpass=80", {"name": "normalize",
                                 "value": -20}]) or @FILE holding either
              --bits B           the samples written: 16 (default), 24, 32, 8 or float
              --block-seconds S  run the sound through the pipeline in blocks of S seconds, as live sound goes (a
                                 whole-recording step waits for the end; the same samples); default 0: all at once
              --no-warm          with -v, the cold run only

        Examples:
          idrak audio process in.wav -o out.wav --pipeline "denoise=1.5, normalize=-20"
          idrak audio process phone.mp3 -o clear.wav --pipeline "highpass=80, dehum=50, presence=3, agc"
          idrak audio process talk.flac -o talk.wav --pipeline "loudness=-16 ceiling=-1" --bits 24
          idrak audio process mic.wav -o live.wav --pipeline "highpass=80, agc, limit" --block-seconds 0.1
          idrak audio process in.wav -o out.wav --pipeline @clean.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out", "--pipeline", "--bits", "--block-seconds"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--no-warm"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string input = context.Argument(0, "IN (the sound file to process, or - for standard input)");
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"audio process reads one file; given {context.Positional.Count}: {string.Join(", ", context.Positional)}.");
        }

        string output = context.Option("--out") ?? throw new UsageException("Missing -o, --out FILE (the sound to write).");
        var pipeline = AudioFiles.Pipeline("--pipeline", context.Option("--pipeline")
            ?? throw new UsageException("Missing --pipeline P (such as \"highpass=80, denoise=1.5, normalize=-20\"; idrak audio processors lists them)."));
        double blockSeconds = AudioFiles.Number(context, "--block-seconds", 0, 0);
        var encoder = AudioFiles.EncoderFor(output);                               // checked before any work is done
        var bits = AudioFiles.Bits(context.Option("--bits"));
        if (input != AudioFiles.StandardInputName && !File.Exists(input))
        {
            throw new UsageException($"Sound file not found: {input}");
        }

        string display = input == AudioFiles.StandardInputName ? "standard input" : Path.GetFileName(input);
        var audio = AudioFiles.Decode(input);
        var (processed, cold) = Process(context, pipeline, audio, blockSeconds, $"processing {display}");
        TimeSpan? warm = null;
        if (context.Verbose && !context.Flag("--no-warm"))
        {
            warm = Process(context, pipeline, audio, blockSeconds, $"processing {display} again (warm)").Elapsed;
        }

        AudioFiles.Save(output, processed, encoder, bits);
        double seconds = audio.Duration.TotalSeconds;
        var steps = new JsonArray();
        foreach (var step in pipeline.Steps)
        {
            var processor = AudioProcessors.Get(step.Name);
            steps.Add((JsonNode)new JsonObject { ["step"] = step.ToString(), ["name"] = step.Name, ["label"] = processor.Label, ["live"] = processor.Live });
            context.Write($"  {step}  ({processor.Label}{(processor.Live ? "" : ", whole recording")})");
        }

        if (pipeline.IsEmpty)
        {
            context.Write("  no steps: the sound as it was decoded");
        }

        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"ran {pipeline.Steps.Count} step(s) over {display} ({AudioFiles.Seconds(seconds)} at {audio.SampleRate} Hz, {audio.Channels} channel(s)) in {cold.TotalMilliseconds:0} ms: "
            + $"real-time factor {Factor(cold, seconds):0.####} ({AudioFiles.Pace(Factor(cold, seconds))}){(blockSeconds > 0 ? $", in blocks of {blockSeconds:0.###} s" : "")}"));
        if (warm is { } w)
        {
            context.Write(string.Create(CultureInfo.InvariantCulture, $"warm: {w.TotalMilliseconds:0} ms, real-time factor {Factor(w, seconds):0.####} ({AudioFiles.Pace(Factor(w, seconds))})"));
        }

        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"wrote {output} ({AudioFiles.Seconds(processed.Duration.TotalSeconds)} at {processed.SampleRate} Hz, {processed.Channels} channel(s), {AudioFiles.BitsName(bits)}-bit)"));
        var json = new JsonObject
        {
            ["in"] = input,
            ["out"] = output,
            ["pipeline"] = pipeline.ToString(),
            ["steps"] = steps,
            ["seconds_in"] = Math.Round(seconds, 4),
            ["seconds_out"] = Math.Round(processed.Duration.TotalSeconds, 4),
            ["rate_in"] = audio.SampleRate,
            ["rate_out"] = processed.SampleRate,
            ["channels_in"] = audio.Channels,
            ["channels_out"] = processed.Channels,
            ["bits"] = AudioFiles.BitsName(bits),
            ["block_seconds"] = blockSeconds,
            ["cold_ms"] = Math.Round(cold.TotalMilliseconds, 3),
            ["real_time_factor"] = Math.Round(Factor(cold, seconds), 6),
        };
        if (warm is { } warmTime)
        {
            json["warm_ms"] = Math.Round(warmTime.TotalMilliseconds, 3);
            json["warm_real_time_factor"] = Math.Round(Factor(warmTime, seconds), 6);
        }

        context.WriteJson(json);
        return ExitCodes.Ok;
    }

    private static double Factor(TimeSpan elapsed, double seconds) => elapsed.TotalSeconds / Math.Max(1e-9, seconds);

    // The sound through the pipeline, timed, with a progress line: the whole recording at once (Apply), or block by block
    // through one opened stage (the blocks given in order, what the stages hold given when flushed).
    private static (AudioData Audio, TimeSpan Elapsed) Process(CommandContext context, AudioPipeline pipeline, AudioData audio, double blockSeconds, string label)
    {
        int block = blockSeconds > 0 ? Math.Max(1, (int)Math.Round(blockSeconds * audio.SampleRate)) : 0;
        int blocks = block > 0 ? (audio.Length + block - 1) / block : 0;
        using var line = new ProgressLine(context, $"{label}: {pipeline}", total: block > 0 ? blocks : null, unit: block > 0 ? ProgressUnit.Items : ProgressUnit.Elapsed);
        long start = Stopwatch.GetTimestamp();
        AudioData result;
        try
        {
            if (block == 0)
            {
                result = pipeline.Apply(audio);
            }
            else
            {
                using var stage = pipeline.Open(audio.SampleRate, audio.Channels);
                var parts = new List<AudioData>(blocks + 1);
                for (int from = 0; from < audio.Length; from += block)
                {
                    var part = stage.Process(audio.Slice(from, Math.Min(block, audio.Length - from)));   // a copy: the stage writes in place
                    if (part.Length > 0)
                    {
                        parts.Add(part);
                    }

                    line.Advance();
                }

                var tail = stage.Flush();
                if (tail.Length > 0)
                {
                    parts.Add(tail);
                }

                result = Join(parts, stage.Channels, stage.SampleRate, stage.Latency);   // the look-ahead off: the file Apply writes
            }
        }
        catch (ArgumentException e)
        {
            line.Erase();
            throw new UsageException($"--pipeline: {e.Message}");                  // a step that cannot run at this rate or channel count
        }

        var elapsed = Stopwatch.GetElapsedTime(start);
        line.Clear();
        return (result, elapsed);
    }

    // Blocks of the same rate and channels, in order, as one recording (planar), each channel copied in one pass.
    // The parts in order as one recording, its first `skip` samples (the stages' shift) left out.
    private static AudioData Join(List<AudioData> parts, int channels, int sampleRate, int skip)
    {
        if (parts.Count == 1 && skip == 0)
        {
            return parts[0];
        }

        int length = Math.Max(0, parts.Sum(p => p.Length) - skip);
        var samples = GC.AllocateUninitializedArray<float>(channels * length);
        for (int c = 0; c < channels; c++)
        {
            int at = c * length, left = skip;
            foreach (var part in parts)
            {
                int from = Math.Min(left, part.Length);
                left -= from;
                part.Channel(c)[from..].CopyTo(samples.AsSpan(at, part.Length - from));
                at += part.Length - from;
            }
        }

        return new AudioData(samples, channels, sampleRate);
    }
}

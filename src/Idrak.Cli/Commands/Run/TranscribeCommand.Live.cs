// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Run;

// idrak transcribe --live (plan 30): the library's LiveTranscriber over a microphone (AudioCaptures), standard input
// (PcmStreamInput) or a file streamed in blocks; the live line shows the open segment, each finished one is printed.
internal sealed partial class TranscribeCommand
{
    private const double FileBlockSeconds = 0.1;

    private static int ListInputs(CommandContext context)
    {
        var captures = new JsonArray();
        foreach (string name in AudioCaptures.Names.Order(StringComparer.Ordinal))
        {
            var capture = AudioCaptures.Get(name);
            bool available = capture.IsAvailable(out string reason);
            var devices = available ? capture.Devices() : [];
            context.Write(available ? $"{name}: {devices.Count} input(s)" : $"{name}: not here ({reason})");
            foreach (var device in devices)
            {
                context.Write($"  {device.Id,-12} {device.Name}{(device.IsDefault ? "  (default)" : "")}");
            }

            captures.Add((JsonNode)new JsonObject
            {
                ["name"] = name, ["available"] = available, ["reason"] = available ? null : reason,
                ["inputs"] = new JsonArray([.. devices.Select(d => (JsonNode)new JsonObject { ["id"] = d.Id, ["name"] = d.Name, ["default"] = d.IsDefault })]),
            });
        }

        context.WriteJson(new JsonObject { ["captures"] = captures });
        return ExitCodes.Ok;
    }

    private static int Live(CommandContext context, string name)
    {
        if (context.Positional.Count > 2)
        {
            throw new UsageException("--live reads one source: nothing (the microphone), - (standard input) or one FILE.");
        }

        string? output = context.Option("--out");
        string format = Format(context, output, toFolder: false);
        var transcription = Options(context, format) with { Timestamps = false, ChunkSeconds = 0 };   // the segments' times are the live ones
        var live = new LiveTranscriptionOptions
        {
            Transcription = transcription,
            StepSeconds = AudioFiles.Number(context, "--step", 1.0, 0.05),
            WindowSeconds = AudioFiles.Number(context, "--window", 20, 0.1),
            SilenceSeconds = AudioFiles.Number(context, "--silence", 0.6, 0.02),
            SilenceDecibels = AudioFiles.Number(context, "--silence-db", 35, 1),
            SilenceFloorDecibels = AudioFiles.Number(context, "--silence-floor-db", -50, -200, -1),
            Agreement = context.IntOption("--agreement", 2) is > 0 and var agreement ? agreement : throw new UsageException("--agreement needs a whole number above 0."),
            Policy = context.Option("--commit") ?? LiveCommitPolicies.Agreement,
        };
        try
        {
            live.Check();
            _ = LiveCommitPolicies.Get(live.Policy);
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            throw new UsageException(e.Message);
        }

        using var input = OpenSource(context);
        live = live with { KeepUp = input is not FileInput && !context.Flag("--every-step") };   // a file, or a recording piped in, at its own pace
        var modelOptions = new SpeechModelOptions { Device = context.Device, Weights = Weights(context.Option("--weights")), Cache = Cache(context.Option("--kv")) };
        var choice = ModelChoices.Choose(context, name);
        string folder = ModelChoices.Resolve(context, choice.Model);
        ISpeechRecognizer recognizer;
        using (var loading = new ProgressLine(context, $"loading {choice.Model} on {context.Device}", unit: ProgressUnit.Elapsed))
        {
            recognizer = SpeechModels.Load(folder, modelOptions);
            loading.Clear();
        }

        using (recognizer)
        using (var interrupt = new Interrupt(context))
        {
            context.Detail($"live from {input.Description}: {input.SampleRate} Hz, {input.Channels} channel(s); a step every {live.StepSeconds:0.##} s");
            var segments = Run(context, recognizer, input, live, interrupt).GetAwaiter().GetResult();
            string text = string.Join(" ", segments.Select(s => s.Text));
            double seconds = segments.Count == 0 ? 0 : segments[^1].End;
            var result = new Transcription(text, transcription.Language, segments,
                new TranscriptionStats(TimeSpan.FromSeconds(seconds), TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero,
                    ComputeResources.GetMemoryUsage(recognizer.Device).Peak, segments.Count, 1));
            var json = TranscriptionJson(input.Description, result);
            json["live"] = true;
            if (output is not null)
            {
                Write(output, Render(format, result, json));
                json["out"] = output;
                context.Write($"wrote {output} ({segments.Count} segment(s), {AudioFiles.Seconds(seconds)})");
            }

            context.WriteJson(new JsonObject { ["model"] = choice.Model, ["device"] = recognizer.Device.ToString(), ["files"] = new JsonArray(json), ["text"] = text });
            return ExitCodes.Ok;
        }
    }

    // The sound: the microphone with no AUDIO, standard input for -, else the file streamed in blocks.
    private static IAudioInput OpenSource(CommandContext context)
    {
        int rate = context.IntOption("--input-rate", 16000) is > 0 and var r ? r : throw new UsageException("--input-rate needs a whole number of Hz above 0.");
        int channels = context.IntOption("--input-channels", 1) is > 0 and var c ? c : throw new UsageException("--input-channels needs a whole number above 0.");
        string? source = context.Positional.Count > 1 ? context.Positional[1] : null;
        if (source == AudioFiles.StandardInputName)
        {
            var format = context.Option("--input-format")?.ToLowerInvariant() switch
            {
                null or "s16le" or "s16" => PcmSampleFormat.S16LE,
                "f32le" or "f32" => PcmSampleFormat.F32LE,
                var other => throw new UsageException($"--input-format takes s16le or f32le, not '{other}' (a WAV header is read whatever it says)."),
            };
            return new PcmStreamInput(Console.OpenStandardInput(), "standard input", format, rate, channels);
        }

        if (source is not null)
        {
            return new FileInput(source, AudioFiles.Decode(source));
        }

        IAudioCapture capture;
        try
        {
            capture = context.Option("--capture") is { } named ? AudioCaptures.Get(named) : AudioCaptures.ForThisMachine();
            if (!capture.IsAvailable(out string reason))
            {
                throw new NotSupportedException($"The {capture.Name} capture cannot record here: {reason}.");
            }

            return capture.Open(new AudioCaptureOptions { Device = context.Option("--input"), SampleRate = rate, Channels = channels });
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException or InvalidOperationException)
        {
            throw new UsageException($"{e.Message} Pipe the sound in instead: {PipeExample()}");
        }
    }

    // A recorder piping 16 kHz mono PCM on this system.
    private static string PipeExample() =>
        OperatingSystem.IsWindows() ? "ffmpeg -f dshow -i audio=\"NAME\" -ac 1 -ar 16000 -f s16le - | idrak transcribe MODEL --live -"
        : OperatingSystem.IsMacOS() ? "ffmpeg -f avfoundation -i :0 -ac 1 -ar 16000 -f s16le - | idrak transcribe MODEL --live -"
        : "arecord -f S16_LE -r 16000 -c 1 -t raw | idrak transcribe MODEL --live -";

    // The live loop: the open segment on the live line, each finished segment printed; Ctrl+C or --timeout stops the input.
    private static async Task<List<TranscriptSegment>> Run(CommandContext context, ISpeechRecognizer recognizer, IAudioInput input, LiveTranscriptionOptions options, Interrupt interrupt)
    {
        var segments = new List<TranscriptSegment>();
        string label = $"live: {input.Description}";
        using var line = new ProgressLine(context, label + "  listening", unit: ProgressUnit.Elapsed);
        line.Report(0);
        var heard = new StringBuilder();
        bool times = context.Flag("--timestamps") || context.Verbose;
        await foreach (var update in LiveTranscriber.TranscribeAsync(recognizer, input.ReadAsync(interrupt.Token), options).ConfigureAwait(false))
        {
            if (update.SegmentEnded)
            {
                heard.Clear();
                var segment = new TranscriptSegment(update.Start, update.End, update.Segment!, 0f, []);
                segments.Add(segment);
                line.Erase();
                if (!context.Json)
                {
                    context.Output.WriteLine(times ? $"[{Clock(update.Start)} - {Clock(update.End)}] {segment.Text}" : segment.Text);
                    context.Output.Flush();
                }
            }
            else if (update.Committed.Length > 0)
            {
                heard.Append(heard.Length > 0 ? " " : "").Append(update.Committed);
            }

            string behind = update.Behind >= options.StepSeconds ? string.Create(CultureInfo.InvariantCulture, $"  ({update.Behind:0.0} s behind)") : "";
            line.Label = $"{label}  {Clock(update.End)}{behind}  {heard}{(update.Tentative.Length > 0 ? (heard.Length > 0 ? " " : "") + "~" + update.Tentative : "")}";
            line.Report(0);
        }

        line.Clear();
        return segments;
    }

    private static string Clock(double seconds) => TimeSpan.FromSeconds(seconds).ToString(seconds >= 3600 ? @"h\:mm\:ss\.f" : @"mm\:ss\.f", CultureInfo.InvariantCulture);

    // A file as live sound: its blocks given in order as fast as they are read (to try the live settings on a recording).
    private sealed class FileInput(string path, AudioData audio) : IAudioInput
    {
        public string Description { get; } = path == AudioFiles.StandardInputName ? "standard input" : Path.GetFileName(path);

        public int SampleRate => audio.SampleRate;

        public int Channels => 1;

        public async IAsyncEnumerable<AudioData> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            int block = Math.Max(1, (int)Math.Round(FileBlockSeconds * audio.SampleRate));
            for (int start = 0; start < mono.Length && !cancellationToken.IsCancellationRequested; start += block)
            {
                yield return new AudioData(mono.AsSpan(start, Math.Min(block, mono.Length - start)).ToArray(), 1, audio.SampleRate);
                await Task.Yield();
            }
        }

        public void Dispose()
        {
        }
    }
}

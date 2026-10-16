// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Commands.Train;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Run;

/// <summary>
/// <c>idrak transcribe MODEL AUDIO...</c> (plan 21, S7): speech to text through the model's registered family
/// (<see cref="SpeechModels"/>), one <see cref="ISpeechRecognizer.Transcribe"/> per file in the order given, as text,
/// JSON, SRT or WebVTT.
/// </summary>
internal sealed class TranscribeCommand : Command
{
    private static readonly string[] Formats = ["text", "json", "srt", "vtt"];

    public override string Name => "transcribe";

    public override string Summary => "Speech to text with a speech recognition model, as text, JSON or subtitles (SRT, WebVTT)";

    public override string Usage => """
        MODEL AUDIO... [options]

        Transcribes each sound file with the model's family (a plug-in given with -P: the library registers none), in the
        order given; a progress line shows the file, the stage and the segment. Sound of any rate and channels is mixed
        to one and resampled to the model's. Without -o the result is printed; with -o it is written: a file for one
        input, else a folder of NAME.txt, .json, .srt or .vtt. -j prints one JSON document (every file's text, language,
        segments and stage times). -v adds each file's stage times (the first file cold, the others warm; one file is
        transcribed again for its warm times) and the device's peak memory. Subtitles take the segments' times, so
        --format srt and vtt ask the model for timestamps.

        Arguments:
          MODEL                  a model folder (config.json, weights, tokenizer, preprocessor_config.json), a hub id
                                 in the cache or an alias
          AUDIO                  sound files (WAV, FLAC; MP3 with its plug-in), folders, globs, or - for standard input

        Options:
          -l, --language CODE    the spoken language (en, ar, ...; default: detected by the model when it can)
              --task T           transcribe (default) or translate (into the family's target language)
              --timestamps       segments with the model's times
              --beam N           beams kept while decoding (1, the default: greedy)
              --length-penalty P the beam search's length penalty (default 1)
              --max-tokens N     new tokens at most per segment (default: the model's longest)
              --chunk-seconds S  seconds of sound decoded at a time (default: the model's segment length)
              --overlap-seconds S  seconds each chunk overlaps the one before with --chunk-seconds (default 1)
              --no-carry         do not give a chunk's text to the next as its prompt
              --prompt TEXT      text given to the model before the first segment (names, spellings)
              --temperature T    the first decoding's temperature (0, the default: greedy or beam; above 0: sampling)
              --fallback T,T...  temperatures tried in turn when a segment's text loops (compresses better than
                                 --compression-ratio-threshold) or the model is unsure of it (mean log-probability
                                 under --logprob-threshold); default 0.2,0.4,0.6,0.8,1.0, as Whisper's own tool
              --no-fallback      decode each segment once (at --temperature)
              --best-of N        samples drawn at a temperature above 0, the likeliest kept (default 5)
              --compression-ratio-threshold C  the text's compression ratio that counts as a loop (default 2.4)
              --logprob-threshold L  the mean log-probability under which a decoding is tried again (default -1)
              --no-speech-threshold P  leave out a segment the model takes for silence: its no-speech probability
                                 above P while unsure of its text (default 0.6; what it writes there is often invented)
              --keep-silence     no no-speech check: every segment's text is kept
              --batch N          segments encoded together at most (default: measured on the device)
              --memory-fraction F  the share of the device's free memory a batch of segments may take (default 0.5)
              --bf16, --no-bf16  run the encoder in bfloat16 where the device has it (the default), or in float32
              --no-timestamp-rules  the model's raw timestamp choices (default: the family's rules over them)
              --no-suppress      the model's raw choices (default: the family's lists of tokens never chosen)
          -w, --weights FORMAT   the projection weights: as-stored (default), f32 or bf16
          -k, --kv FORMAT        the decoder's key/value cache: float32 (default), int8 or bf16
          -o, --out FILE|DIR     write the result: FILE for one input, else a folder of NAME.EXT
              --format F         text (default), json, srt or vtt (also taken from -o FILE's extension)
              --no-warm          with -v, report the cold times only (one file is not transcribed again)

        Examples:
          idrak transcribe -P Idrak.WhisperSpeech.dll ./whisper-small talk.wav
          idrak transcribe -P Idrak.WhisperSpeech.dll ./whisper-small lecture.flac -l ar --timestamps -o lecture.srt
          idrak transcribe -P Idrak.WhisperSpeech.dll -P Idrak.Mp3Audio.dll ./whisper-small recordings/ -o texts/ --format vtt -d cuda
          idrak transcribe -P Idrak.WhisperSpeech.dll ./whisper-small talk.wav --beam 4 -j -v
          cat talk.wav | idrak transcribe -P Idrak.WhisperSpeech.dll ./whisper-small -
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--language", "--task", "--beam", "--length-penalty", "--max-tokens", "--chunk-seconds", "--overlap-seconds", "--prompt", "--temperature", "--batch",
        "--memory-fraction", "--weights", "--kv", "--out", "--fallback", "--best-of", "--compression-ratio-threshold", "--logprob-threshold", "--no-speech-threshold",
    ];

    public override IReadOnlyCollection<string> Flags { get; } =
        ["--timestamps", "--no-carry", "--bf16", "--no-bf16", "--no-timestamp-rules", "--no-suppress", "--no-warm", "--no-fallback", "--keep-silence"];

    /// <summary>
    /// The command's defaults: the library's, with OpenAI Whisper's guards on (the temperature fallback and the
    /// no-speech check), which the library leaves off.
    /// </summary>
    internal static TranscriptionOptions Defaults => new() { TemperatureFallback = [0.2f, 0.4f, 0.6f, 0.8f, 1f], NoSpeechThreshold = 0.6f };

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string> { ["-l"] = "--language", ["-o"] = "--out", ["-w"] = "--weights", ["-k"] = "--kv" };

    public override IReadOnlyCollection<string> OwnFormats => ["srt", "vtt"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a speech recognition model folder, a hub id in the cache or an alias)");
        if (context.Positional.Count < 2)
        {
            throw new UsageException("Missing AUDIO (sound files, folders, globs, or - for standard input).");
        }

        var files = AudioFiles.Expand(context.Positional.Skip(1), standardInput: true);
        if (files.Count == 0)
        {
            throw new UsageException($"No sound files in {string.Join(", ", context.Positional.Skip(1))} (the registered codecs read {string.Join(", ", AudioCodecs.Extensions)}).");
        }

        string? output = context.Option("--out");
        bool toFolder = output is not null && (files.Count > 1 || Directory.Exists(output) || Path.EndsInDirectorySeparator(output) || Path.GetExtension(output).Length == 0);
        string format = Format(context, output, toFolder);
        var options = Options(context, format);
        var modelOptions = new SpeechModelOptions { Device = context.Device, Weights = Weights(context.Option("--weights")), Cache = Cache(context.Option("--kv")) };
        var choice = ModelChoices.Choose(context, name);
        string folder = ModelChoices.Resolve(context, choice.Model);
        try
        {
            ISpeechRecognizer recognizer;
            using (var loading = new ProgressLine(context, $"loading {choice.Model} on {context.Device}", unit: ProgressUnit.Elapsed))
            {
                recognizer = SpeechModels.Load(folder, modelOptions);
                loading.Clear();
            }

            using (recognizer)
            {
                context.Detail($"loaded {choice.Model} on {recognizer.Device}: {recognizer.Languages.Count} language(s)");
                var results = new JsonArray();
                var printed = new StringBuilder();
                for (int i = 0; i < files.Count; i++)
                {
                    string file = files[i];
                    var audio = AudioFiles.Decode(file);
                    var result = Transcribe(context, recognizer, audio, options, $"{i + 1}/{files.Count} {Display(file)}");
                    if (context.Verbose && !context.Json)
                    {
                        context.ErrorOutput.WriteLine(StageLine(i == 0 ? "cold" : "warm", Display(file), result.Stats));
                    }

                    if (i == 0 && files.Count == 1 && context.Verbose && !context.Flag("--no-warm"))
                    {
                        var warm = Transcribe(context, recognizer, audio, options, $"again (warm) {Display(file)}");
                        if (!context.Json)
                        {
                            context.ErrorOutput.WriteLine(StageLine("warm", Display(file), warm.Stats));
                        }
                    }

                    var json = TranscriptionJson(file, result);
                    results.Add((JsonNode)json);
                    string content = Render(format, result, json);
                    if (output is not null)
                    {
                        string path = toFolder ? Path.Combine(output, $"{(file == AudioFiles.StandardInputName ? "stdin" : Path.GetFileNameWithoutExtension(file))}.{Extension(format)}") : output;
                        Write(path, content);
                        json["out"] = path;
                        context.Write($"wrote {path} ({AudioFiles.Seconds(result.Stats.AudioDuration.TotalSeconds)} of sound, {result.Segments.Count} segment(s){(result.Language is null ? "" : $", {result.Language}")})");
                    }
                    else if (!context.Json)
                    {
                        if (files.Count > 1 && format == "text")
                        {
                            printed.Append("# ").Append(Display(file)).Append('\n');
                        }

                        printed.Append(content);
                        context.Output.Write(content);
                        if (!content.EndsWith('\n'))
                        {
                            context.Output.WriteLine();
                        }
                    }
                }

                context.Log(printed.ToString());
                var document = new JsonObject { ["model"] = choice.Model, ["device"] = recognizer.Device.ToString(), ["files"] = results };
                if (results.Count == 1)
                {
                    document["text"] = results[0]!["text"]?.DeepClone();
                    document["language"] = results[0]!["language"]?.DeepClone();
                }

                context.WriteJson(document);
                return ExitCodes.Ok;
            }
        }
        catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }
    }

    // The format: --format, else -j's json, else -o FILE's extension, else text.
    private static string Format(CommandContext context, string? output, bool toFolder)
    {
        string? given = context.Option("--format");
        if (given is not null)
        {
            return Formats.Contains(given, StringComparer.Ordinal) ? given : throw new UsageException($"transcribe writes --format text, json, srt or vtt, not '{given}'.");
        }

        if (context.Json)
        {
            return "json";
        }

        string extension = output is null || toFolder ? "" : Path.GetExtension(output).TrimStart('.').ToLowerInvariant();
        return extension switch { "json" => "json", "srt" => "srt", "vtt" => "vtt", _ => "text" };
    }

    private static string Extension(string format) => format == "text" ? "txt" : format;

    private static TranscriptionOptions Options(CommandContext context, string format)
    {
        var d = Defaults;
        if (context.Flag("--bf16") && context.Flag("--no-bf16"))
        {
            throw new UsageException("Give --bf16 or --no-bf16, not both.");
        }

        string task = context.Option("--task") ?? d.Task;
        if (task is not ("transcribe" or "translate"))
        {
            throw new UsageException($"--task takes transcribe or translate, not '{task}'.");
        }

        return d with
        {
            Language = context.Option("--language"),
            Task = task,
            Timestamps = context.Flag("--timestamps") || format is "srt" or "vtt",
            TimestampRules = !context.Flag("--no-timestamp-rules"),
            Beams = context.IntOption("--beam", d.Beams) is > 0 and var beams ? beams : throw new UsageException("--beam needs a whole number above 0."),
            LengthPenalty = (float)AudioFiles.Number(context, "--length-penalty", d.LengthPenalty),
            MaxTokens = context.IntOption("--max-tokens", d.MaxTokens) is >= 0 and var tokens ? tokens : throw new UsageException("--max-tokens needs a whole number, 0 or more."),
            ChunkSeconds = AudioFiles.Number(context, "--chunk-seconds", d.ChunkSeconds, 0),
            OverlapSeconds = AudioFiles.Number(context, "--overlap-seconds", d.OverlapSeconds, 0),
            CarryText = !context.Flag("--no-carry"),
            Prompt = context.Option("--prompt"),
            Temperature = (float)AudioFiles.Number(context, "--temperature", d.Temperature, 0),
            TemperatureFallback = context.Flag("--no-fallback") ? [] : context.Option("--fallback") is { } ladder ? Temperatures(ladder) : d.TemperatureFallback,
            BestOf = context.IntOption("--best-of", d.BestOf) is > 0 and var bestOf ? bestOf : throw new UsageException("--best-of needs a whole number above 0."),
            CompressionRatioThreshold = (float)AudioFiles.Number(context, "--compression-ratio-threshold", d.CompressionRatioThreshold, 0),
            LogProbThreshold = (float)AudioFiles.Number(context, "--logprob-threshold", d.LogProbThreshold),
            NoSpeechThreshold = context.Flag("--keep-silence") ? null : (float)AudioFiles.Number(context, "--no-speech-threshold", d.NoSpeechThreshold ?? 0.6, 0, 1),
            Seed = context.IntOption("--seed", d.Seed),
            MaxBatch = context.IntOption("--batch", d.MaxBatch) is >= 0 and var batch ? batch : throw new UsageException("--batch needs a whole number, 0 or more."),
            MemoryFraction = AudioFiles.Number(context, "--memory-fraction", d.MemoryFraction, double.Epsilon, 1),
            BFloat16 = !context.Flag("--no-bf16"),
            SuppressTokens = !context.Flag("--no-suppress"),
        };
    }

    // "0.2,0.4,1" → the temperatures, each 0 or more.
    private static float[] Temperatures(string text)
    {
        var values = new List<float>();
        foreach (string part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            values.Add(float.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out float t) && t >= 0 && float.IsFinite(t) ? t
                : throw new UsageException($"--fallback takes temperatures 0 or more separated by commas (0.2,0.4,0.6), not '{part}'."));
        }

        return [.. values];
    }

    private static EncoderWeights Weights(string? text) => text?.ToLowerInvariant() switch
    {
        null or "as-stored" or "stored" => EncoderWeights.AsStored,
        "f32" or "float32" => EncoderWeights.Float32,
        "bf16" or "bfloat16" => EncoderWeights.BFloat16,
        _ => throw new UsageException($"--weights takes as-stored, f32 or bf16, not '{text}'."),
    };

    private static KeyValueFormat Cache(string? text) => text?.ToLowerInvariant() switch
    {
        null or "float32" or "f32" => KeyValueFormat.Float32,
        "int8" => KeyValueFormat.Int8,
        "bf16" or "bfloat16" => KeyValueFormat.BFloat16,
        _ => throw new UsageException($"--kv takes float32, int8 or bf16, not '{text}'."),
    };

    private static string Display(string file) => file == AudioFiles.StandardInputName ? "standard input" : Path.GetFileName(file);

    // One call with a progress line: the file, the stage, the segments done of all.
    private static Transcription Transcribe(CommandContext context, ISpeechRecognizer recognizer, AudioData audio, TranscriptionOptions options, string label)
    {
        using var line = new ProgressLine(context, label, unit: ProgressUnit.Items);
        var result = recognizer.Transcribe(audio, options, new Reports(line, label), context.TimeoutToken);
        line.Clear();
        return result;
    }

    // Progress reported on the calling thread (Progress<T> would post it to the thread pool, after the call returned).
    private sealed class Reports(ProgressLine line, string label) : IProgress<TranscriptionProgress>
    {
        public void Report(TranscriptionProgress value)
        {
            line.Label = $"{label}  {value.Stage}  segment";
            line.Total = value.Segments;
            line.Report(value.Segment);
        }
    }

    private static string StageLine(string label, string file, TranscriptionStats s) => string.Create(CultureInfo.InvariantCulture,
        $"{label} {file}: {AudioFiles.Seconds(s.AudioDuration.TotalSeconds)} of sound in {s.Total.TotalMilliseconds:0} ms (decode {s.Decode.TotalMilliseconds:0}, "
        + $"features {s.Features.TotalMilliseconds:0}, encoder {s.Encoder.TotalMilliseconds:0}, decoder {s.Decoder.TotalMilliseconds:0} ms), "
        + $"x{s.Total.TotalSeconds / Math.Max(1e-9, s.AudioDuration.TotalSeconds):0.000} real time, {s.Segments} segment(s) in batches of {s.Batch}, "
        + $"peak device memory {Units.Bytes(s.PeakDeviceBytes)}");

    /// <summary>A transcription as JSON: the file, the text, the language, the segments and the stage times.</summary>
    internal static JsonObject TranscriptionJson(string file, Transcription t) => new()
    {
        ["file"] = file,
        ["text"] = t.Text.Trim(),
        ["language"] = t.Language,
        ["segments"] = new JsonArray([.. t.Segments.Select(s => (JsonNode)new JsonObject
        {
            ["start"] = Math.Round(s.Start, 3), ["end"] = Math.Round(s.End, 3), ["text"] = s.Text.Trim(), ["avg_logprob"] = s.AverageLogProb,
            ["tokens"] = new JsonArray([.. s.Tokens.Select(id => (JsonNode)id)]),
        })]),
        ["stats"] = new JsonObject
        {
            ["audio_seconds"] = Math.Round(t.Stats.AudioDuration.TotalSeconds, 4),
            ["decode_ms"] = Math.Round(t.Stats.Decode.TotalMilliseconds, 3),
            ["features_ms"] = Math.Round(t.Stats.Features.TotalMilliseconds, 3),
            ["encoder_ms"] = Math.Round(t.Stats.Encoder.TotalMilliseconds, 3),
            ["decoder_ms"] = Math.Round(t.Stats.Decoder.TotalMilliseconds, 3),
            ["peak_device_bytes"] = t.Stats.PeakDeviceBytes,
            ["segments"] = t.Stats.Segments,
            ["batch"] = t.Stats.Batch,
        },
    };

    // The result in its format.
    private static string Render(string format, Transcription t, JsonObject json) => format switch
    {
        "json" => json.ToJsonString(CommandContext.JsonOutput) + "\n",
        "srt" => Subtitles(t, vtt: false),
        "vtt" => Subtitles(t, vtt: true),
        _ => t.Text.Trim() + "\n",
    };

    /// <summary>
    /// The segments as SubRip (numbered cues, HH:MM:SS,mmm) or WebVTT (a WEBVTT header, HH:MM:SS.mmm); segments without
    /// text are left out.
    /// </summary>
    internal static string Subtitles(Transcription transcription, bool vtt)
    {
        var text = new StringBuilder(vtt ? "WEBVTT\n\n" : "");
        int cue = 0;
        foreach (var segment in transcription.Segments)
        {
            string words = segment.Text.Trim();
            if (words.Length == 0)
            {
                continue;
            }

            if (!vtt)
            {
                text.Append(++cue).Append('\n');
            }

            text.Append(Time(segment.Start, vtt)).Append(" --> ").Append(Time(Math.Max(segment.Start, segment.End), vtt)).Append('\n').Append(words).Append("\n\n");
        }

        return text.ToString();
    }

    private static string Time(double seconds, bool vtt)
    {
        long ms = (long)Math.Round(Math.Max(0, seconds) * 1000);
        return string.Create(CultureInfo.InvariantCulture, $"{ms / 3_600_000:00}:{ms / 60_000 % 60:00}:{ms / 1000 % 60:00}{(vtt ? '.' : ',')}{ms % 1000:000}");
    }

    // UTF-8 without a BOM, whatever the console's code page.
    private static void Write(string path, string content)
    {
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        File.WriteAllText(path, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }
}

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Commands.Run;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data mel FILE... -o DIR</c> (plan 21, S7): a model's input features of sound files, written as NumPy .npy
/// files with the options they were taken with (features.json), for inspection and comparison with another
/// implementation. The features are a registered extractor's (<see cref="AudioFeatureExtractors"/>, "log-mel" built in).
/// </summary>
internal sealed class DataMelCommand : Command
{
    /// <summary>The options' file name.</summary>
    public const string OptionsFile = "features.json";

    public override string Name => "data mel";

    public override string Summary => "Log-mel features of sound files as .npy (a model's input features), with the options used";

    public override string Usage => """
        FILE... -o DIR [--preprocessor preprocessor_config.json | feature options] [options]

        Computes the features of each file (mixed to one channel and resampled to the features' rate) on the device and
        writes NAME.npy (float32, [frames, mels], frames as rows; --transpose writes [mels, frames] as transformers'
        input_features) and features.json (the extractor and every option). The options are a model's
        preprocessor_config.json (--preprocessor FILE, or the model folder holding it), then the options below over them.

        Arguments:
          FILE                   sound files (WAV, FLAC; MP3 with its plug-in), folders or globs

        Options:
          -o, --out DIR          the folder to write
              --preprocessor F   a preprocessor_config.json (sampling_rate, n_fft, hop_length, feature_size,
                                 chunk_length), or a model folder that has one
              --extractor NAME   the registered feature extractor (default log-mel)
              --sample-rate R    the features' rate (default 16000)
              --window-size N    the frame's samples, n_fft (default 400)
              --hop-size N       samples between frames (default 160)
              --window W         hann (default), hamming or rectangular
              --no-center        frames start at the first sample (default: centred, the sound reflect-padded)
              --mels M           mel bands (default 80)
              --fmin F           the filterbank's lowest frequency, Hz (default 0)
              --fmax F           its highest, Hz (default: half the rate)
              --mel-scale S      slaney (default) or htk
              --mel-norm N       slaney (default) or none
              --log-floor F      the least power before the log (default 1e-10)
              --dynamic-range D  log values more than D below the largest are raised to it (default 8; none for none)
              --normalize O,S    (x + O) * S after the log (default 4,0.25; none for none)
              --pad-seconds S    pad or cut each file to S seconds (default 30; 0 for the sound's own length)
              --stft-path P      auto (default), host or composed (IDRAK_STFT_PATH overrides auto)
              --transpose        write [mels, frames]

        Examples:
          idrak data mel talk.wav -o mels --preprocessor ./whisper-small
          idrak data mel clips/ -o mels --mels 128 --pad-seconds 0 -d cuda
          idrak data mel talk.wav -o mels --preprocessor preprocessor_config.json --transpose -j
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--out", "--preprocessor", "--extractor", "--sample-rate", "--window-size", "--hop-size", "--window", "--mels", "--fmin", "--fmax", "--mel-scale", "--mel-norm",
        "--log-floor", "--dynamic-range", "--normalize", "--pad-seconds", "--stft-path",
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--no-center", "--transpose"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Missing FILE (the sound files).");
        }

        string output = context.Option("--out") ?? throw new UsageException("Missing -o, --out DIR (the folder to write).");
        var files = AudioFiles.Expand(context.Positional);
        if (files.Count == 0)
        {
            throw new UsageException($"No sound files in {string.Join(", ", context.Positional)} (the registered codecs read {string.Join(", ", AudioCodecs.Extensions)}).");
        }

        var options = Options(context);
        string name = context.Option("--extractor") ?? "log-mel";
        IAudioFeatureExtractor extractor;
        try
        {
            extractor = AudioFeatureExtractors.Create(name, options);
            _ = extractor.FramesFor(options.SampleRate);                       // the options are checked now
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException)
        {
            throw new UsageException(e.Message);
        }

        bool transpose = context.Flag("--transpose");
        Directory.CreateDirectory(output);
        var device = context.Device;
        var written = new JsonArray();
        using (var line = new ProgressLine(context, $"features on {device}", total: files.Count, unit: ProgressUnit.Items))
        {
            foreach (string file in files)
            {
                line.Label = $"features of {Path.GetFileName(file)} on {device}";
                var audio = AudioFiles.Decode(file);
                using var features = extractor.Extract(audio, device);             // [frames, mels]
                int frames = features.Shape[0], mels = features.Shape[1];
                var values = features.ToArray();
                if (transpose)
                {
                    values = Transpose(values, frames, mels);
                }

                string path = Path.Combine(output, Path.GetFileNameWithoutExtension(file) + ".npy");
                NpyFile.Write(path, values, transpose ? [mels, frames] : [frames, mels]);
                written.Add((JsonNode)new JsonObject
                {
                    ["file"] = file, ["out"] = path, ["frames"] = frames, ["mels"] = mels, ["seconds"] = Math.Round(audio.Duration.TotalSeconds, 4),
                });
                context.Detail(string.Create(CultureInfo.InvariantCulture, $"{file}: [{(transpose ? $"{mels}, {frames}" : $"{frames}, {mels}")}] to {path}"));
                line.Advance();
            }

            line.Clear();
        }

        var record = options.ToJson();
        record["extractor"] = extractor.Name;
        record["layout"] = transpose ? "mels, frames" : "frames, mels";
        File.WriteAllText(Path.Combine(output, OptionsFile), record.ToJsonString(CommandContext.JsonOutput) + "\n");
        context.Write($"wrote the {extractor.Name} features of {files.Count} file(s) to {output} ({options.Mels} mels, hop {options.HopSize} at {options.SampleRate} Hz; options in {OptionsFile})");
        context.WriteJson(new JsonObject { ["out"] = output, ["extractor"] = extractor.Name, ["options"] = record, ["files"] = written });
        return ExitCodes.Ok;
    }

    // The preprocessor_config.json's options, then the command line's over them.
    private static AudioFeatureOptions Options(CommandContext context)
    {
        var options = new AudioFeatureOptions();
        if (context.Option("--preprocessor") is { } config)
        {
            string file = Directory.Exists(config) ? Path.Combine(config, "preprocessor_config.json") : config;
            if (!File.Exists(file))
            {
                throw new UsageException($"No preprocessor_config.json at {file}.");
            }

            options = AudioFeatureOptions.FromConfig(JsonNode.Parse(File.ReadAllText(file)) as JsonObject ?? throw new UsageException($"{file} is not a JSON object."));
        }

        int Int(string option, int fallback) => context.IntOption(option, fallback) is > 0 and var v ? v : throw new UsageException($"{option} needs a whole number above 0.");
        double pad = AudioFiles.Number(context, "--pad-seconds", options.PadSeconds ?? 0, 0);
        return options with
        {
            SampleRate = Int("--sample-rate", options.SampleRate),
            WindowSize = Int("--window-size", options.WindowSize),
            HopSize = Int("--hop-size", options.HopSize),
            Window = context.Option("--window") ?? options.Window,
            Center = options.Center && !context.Flag("--no-center"),
            Mels = Int("--mels", options.Mels),
            MinFrequency = (float)AudioFiles.Number(context, "--fmin", options.MinFrequency, 0),
            MaxFrequency = context.Option("--fmax") is null ? options.MaxFrequency : (float)AudioFiles.Number(context, "--fmax", 0, double.Epsilon),
            MelScale = context.Option("--mel-scale") ?? options.MelScale,
            MelNorm = context.Option("--mel-norm") ?? options.MelNorm,
            LogFloor = (float)AudioFiles.Number(context, "--log-floor", options.LogFloor, double.Epsilon),
            DynamicRange = context.Option("--dynamic-range") is "none" ? null : context.Option("--dynamic-range") is null ? options.DynamicRange : (float)AudioFiles.Number(context, "--dynamic-range", 0, 0),
            Normalize = Normalize(context.Option("--normalize"), options.Normalize),
            PadSeconds = pad > 0 ? pad : null,
            Path = context.Option("--stft-path") ?? options.Path,
        };
    }

    private static (float Offset, float Scale)? Normalize(string? text, (float, float)? fallback)
    {
        if (text is null)
        {
            return fallback;
        }

        if (text == "none")
        {
            return null;
        }

        var parts = text.Split(',', StringSplitOptions.TrimEntries);
        return parts.Length == 2 && float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out float offset)
                                 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float scale)
            ? (offset, scale)
            : throw new UsageException($"--normalize takes OFFSET,SCALE (such as 4,0.25) or none, not '{text}'.");
    }

    // [rows, columns] to [columns, rows], walking the source in its order.
    private static float[] Transpose(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }
}

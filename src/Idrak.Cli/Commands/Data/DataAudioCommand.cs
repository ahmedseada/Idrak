// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data audio FILE|DIR... -o DIR</c> (plan 21, S7): sound files converted (channels mixed, resampled, written as
/// WAV or any registered encoder's format), optionally cut into pieces by a registered splitter
/// (<see cref="AudioSplitters"/>: at pauses or fixed lengths), with a manifest.jsonl a speech data source reads, and
/// statistics of the lengths, rates and channels.
/// </summary>
internal sealed class DataAudioCommand : Command
{
    /// <summary>The manifest's file name.</summary>
    public const string ManifestFile = "manifest.jsonl";

    public override string Name => "data audio";

    public override string Summary => "Convert, resample and split sound files into a folder of WAV files with a manifest.jsonl";

    public override string Usage => """
        FILE|DIR... -o DIR [options]

        Decodes each file (WAV, FLAC; MP3 with its plug-in), mixes its channels to one with --mono, resamples it with
        --rate, cuts it into pieces with --split, and writes every piece (NAME.wav, or NAME-0001.wav ... when split; a
        folder's files keep their subfolders) with a line of manifest.jsonl: name, seconds, rate, channels, source,
        start and end in the source, and text when --transcripts has the file's (a split file's pieces get none: its
        text is the whole file's). --stats prints the lengths (with a histogram), rates and channels, of the inputs
        alone without -o. --align MODEL then writes alignments.jsonl beside the manifest: where each label of every
        piece's text is spoken, from a CTC speech model's best paths (the durations a synthesizer trains on, idrak
        pretrain --durations); given speech data instead of sounds to convert (transcripts files, .jsonl, or folders and
        zips of name.wav beside name.txt, with no conversion option), it aligns those utterances as the trainers read
        them (--audio DIR|ZIP where a transcripts file's sounds are) and converts nothing; -o is then the folder
        alignments.jsonl goes in (default: the folder given, or the transcripts file's).

        Arguments:
          FILE|DIR               sound files, folders (all subfolders) or globs

        Options:
          -o, --out DIR          the folder to write (the pieces and manifest.jsonl)
              --rate R           resample to R Hz (default: each file's own rate)
              --mono             mix the channels to one (their mean)
              --bits B           the samples written: 16 (default), 24, 32, 8 or float
              --type EXT         the files written: wav (default), or any extension a registered encoder writes
              --split S          none (default), silence (the pieces between pauses), seconds (pieces of --max-seconds)
                                 or a registered splitter's name
              --silence-db D     a frame more than D dB below the loudest is silence (default 40)
              --min-silence S    a pause of at least S seconds separates two pieces (default 0.3)
              --pad-seconds S    silence kept on both sides of a piece (default 0.1)
              --hop-seconds S    the frames the level is measured over (default 0.01)
              --min-seconds S    leave out pieces shorter than S seconds (default 0: none)
              --max-seconds S    the longest piece: silence cuts a longer one at its quietest frame; seconds cuts
                                 pieces of S (default 0: no limit)
              --resample-width N the resampler's kernel half-width in zero crossings (default 6)
              --rolloff F        the resampler's cutoff as a share of the lower rate's Nyquist frequency (default 0.99)
              --clean STAGES     every piece cleaned before it is written, comma-separated, in order: highpass (DC and
                                 rumble out), denoise (the noise floor gated off), trim (silence at the ends cut),
                                 normalize (speech levelled), or a registered cleaner's name (default none)
              --clean-settings F the stages' settings as JSON, a file or inline (default the library's)
              --transcripts FILE the texts: JSON Lines rows with "file" (or "audio", "path", "name", "id") and "text"
                                 (or "transcript", "sentence"), or lines of NAME<tab>TEXT or NAME|TEXT; matched by file
                                 name, then by name without extension
              --stats            print the lengths, rates and channels (without -o: of the inputs, nothing is written)
              --align MODEL      write alignments.jsonl from a CTC speech model (IdrakAudio's, or a -P plug-in's)
              --align-seconds S  audio seconds in one pass of the model while aligning (default 60)
              --audio DIR|ZIP    where the sounds of transcripts files given to --align are (default: beside each file)
              --source KIND      the speech data's kind for --align (transcripts or audio-folder; told from the paths)

        Examples:
          idrak data audio recordings/ -o clips --rate 16000 --mono
          idrak data audio rec/ -o voice-clips --rate 22050 --mono --split silence --min-seconds 3
          idrak data audio talk.flac -o pieces --split seconds --max-seconds 30 --transcripts texts.jsonl
          idrak data audio podcasts/*.mp3 -o wav --rate 16000 --mono --bits 16
          idrak data audio corpus/ --stats
          idrak data audio rec/ -o clips --rate 16000 --mono --transcripts texts.tsv --align my-ctc
          idrak data audio train.jsonl --audio clips.zip --align my-ctc -o aligned
          idrak data audio pieces/ --align my-ctc
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--out", "--rate", "--bits", "--type", "--split", "--silence-db", "--min-silence", "--pad-seconds", "--hop-seconds", "--min-seconds", "--max-seconds",
        "--resample-width", "--rolloff", "--clean", "--clean-settings", "--transcripts", "--align", "--align-seconds", "--audio", "--source",
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--mono", "--stats"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("Missing FILE|DIR (the sound files to convert).");
        }

        string? output = context.Option("--out");
        bool stats = context.Flag("--stats");
        if (context.Option("--align") is { } aligner && AlignsOnly(context))
        {
            // Speech data as the trainers read it (transcripts files, folders or zips of pairs), aligned, nothing converted.
            var paths = context.Positional.Select(Path.GetFullPath).ToList();
            var given = new SpeechSource(context.Option("--source") ?? Idrak.Cli.Commands.Train.SpeechPretraining.Kind(paths), paths)
            {
                Audio = context.Option("--audio") is { } a ? Path.GetFullPath(a) : null,
            };
            string folder = output ?? (paths.Count == 1 && Directory.Exists(paths[0]) ? paths[0] : Path.GetDirectoryName(paths[0])!);
            var aligned = Align(context, aligner, given, folder);
            context.WriteJson(new JsonObject { ["out"] = folder, ["source"] = given.Kind, ["alignments"] = aligned });
            return ExitCodes.Ok;
        }

        if (output is null && !stats)
        {
            throw new UsageException("Missing -o, --out DIR (the folder to write), or --stats to describe the files only.");
        }

        if ((context.Option("--audio") is not null || context.Option("--source") is not null) && context.Option("--align") is null)
        {
            throw new UsageException("--audio and --source describe speech data given to --align; give --align MODEL with transcripts or a folder of name.wav + name.txt.");
        }

        var inputs = Inputs(context.Positional);
        if (inputs.Count == 0)
        {
            throw new UsageException($"No sound files in {string.Join(", ", context.Positional)} (the registered codecs read {string.Join(", ", AudioCodecs.Extensions)}).");
        }

        int? rate = context.Option("--rate") is null ? null : context.IntOption("--rate", 0) is > 0 and var r ? r : throw new UsageException("--rate needs a whole number of Hz above 0.");
        bool mono = context.Flag("--mono");
        var bits = AudioFiles.Bits(context.Option("--bits"));
        string extension = "." + (context.Option("--type") ?? "wav").TrimStart('.').ToLowerInvariant();
        var encoder = output is null ? null : AudioFiles.EncoderFor("x" + extension);
        string split = context.Option("--split") ?? "none";
        IAudioSplitter? splitter;
        try
        {
            splitter = split == "none" ? null : AudioSplitters.Get(split);
        }
        catch (NotSupportedException e)
        {
            throw new UsageException($"--split: {e.Message} (or none)");
        }

        var defaults = new AudioSplitOptions();
        var splitOptions = new AudioSplitOptions
        {
            SilenceDecibels = (float)AudioFiles.Number(context, "--silence-db", defaults.SilenceDecibels, double.Epsilon),
            MinSilenceSeconds = AudioFiles.Number(context, "--min-silence", defaults.MinSilenceSeconds, 0),
            PadSeconds = AudioFiles.Number(context, "--pad-seconds", defaults.PadSeconds, 0),
            HopSeconds = AudioFiles.Number(context, "--hop-seconds", defaults.HopSeconds, double.Epsilon),
            MinSeconds = AudioFiles.Number(context, "--min-seconds", defaults.MinSeconds, 0),
            MaxSeconds = AudioFiles.Number(context, "--max-seconds", defaults.MaxSeconds, 0),
        };
        try
        {
            splitOptions.Check();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        if (split == AudioSplitters.Seconds && !(splitOptions.MaxSeconds > 0))
        {
            throw new UsageException("--split seconds needs --max-seconds S (the length of a piece).");
        }

        var resample = new ResampleOptions
        {
            Width = context.IntOption("--resample-width", new ResampleOptions().Width) is > 0 and var w ? w : throw new UsageException("--resample-width needs a whole number above 0."),
            Rolloff = AudioFiles.Number(context, "--rolloff", new ResampleOptions().Rolloff, double.Epsilon, 1),
        };
        var transcripts = context.Option("--transcripts") is { } transcriptFile ? Transcripts(transcriptFile) : null;
        IAudioCleaner[] cleaners;
        try
        {
            cleaners = [.. AudioFiles.CleaningStages(context, []).Select(AudioCleaners.Get)];
        }
        catch (NotSupportedException e)
        {
            throw new UsageException($"--clean: {e.Message}");
        }

        var cleaning = AudioFiles.CleaningSettings(context, new AudioCleaningOptions());

        var sources = new List<(string Path, double Seconds, int Rate, int Channels)>();
        var pieces = new List<(string Name, double Seconds, int Rate, int Channels)>();
        int attached = 0, unattached = 0, dropped = 0;
        StreamWriter? manifest = null;
        if (output is not null)
        {
            Directory.CreateDirectory(output);
            manifest = new StreamWriter(Path.Combine(output, ManifestFile), append: false, new UTF8Encoding(false));
        }

        try
        {
            using var line = new ProgressLine(context, output is null ? "reading" : $"converting into {output}", total: inputs.Count, unit: ProgressUnit.Items);
            foreach (var (path, relative) in inputs)
            {
                line.Label = $"{(output is null ? "reading" : "converting")} {relative}";
                if (output is null)
                {
                    // The statistics need the header only (the length, rate and channels), not the samples.
                    var info = AudioCodecs.ReadInfo(path) ?? throw new InvalidDataException($"{path}: not an audio format of the registered codecs ({string.Join(", ", AudioCodecs.Names)}).");
                    sources.Add((path, info.Samples >= 0 ? (double)info.Samples / info.SampleRate : AudioFiles.Decode(path).Duration.TotalSeconds, info.SampleRate, info.Channels));
                    line.Advance();
                    continue;
                }

                var audio = AudioFiles.Decode(path);
                sources.Add((path, audio.Duration.TotalSeconds, audio.SampleRate, audio.Channels));
                if (mono && audio.Channels > 1)
                {
                    var mixed = GC.AllocateUninitializedArray<float>(audio.Length);
                    audio.ToMono(mixed);
                    audio = new AudioData(mixed, 1, audio.SampleRate);
                }

                if (rate is { } to && to != audio.SampleRate)
                {
                    audio = Resampler.Resample(audio, to, resample);
                }

                IReadOnlyList<AudioPiece> cuts = splitter is null ? [new AudioPiece(0, audio.Length)] : splitter.Split(audio, splitOptions);
                if (splitter is not null && cuts.Count == 0)
                {
                    dropped++;
                }

                string stem = Path.ChangeExtension(relative, null);
                string? text = transcripts is null ? null : Text(transcripts, relative);
                bool whole = splitter is null || cuts.Count == 1 && cuts[0].Start == 0 && cuts[0].Length == audio.Length;
                if (text is not null)
                {
                    if (whole)
                    {
                        attached++;
                    }
                    else
                    {
                        unattached++;
                    }
                }

                for (int i = 0; i < cuts.Count; i++)
                {
                    var cut = cuts[i];
                    string name = (splitter is null ? stem : string.Create(CultureInfo.InvariantCulture, $"{stem}-{i + 1:D4}")) + extension;
                    var piece = cut.Start == 0 && cut.Length == audio.Length ? audio : audio.Slice(cut.Start, cut.Length);
                    foreach (var cleaner in cleaners)
                    {
                        piece = cleaner.Clean(piece, cleaning);                 // start and end below stay the piece's cut in the source
                    }

                    AudioFiles.Save(Path.Combine(output, name), piece, encoder!, bits);
                    double seconds = piece.Duration.TotalSeconds;
                    pieces.Add((name, seconds, piece.SampleRate, piece.Channels));
                    var row = new JsonObject
                    {
                        ["name"] = name.Replace('\\', '/'),
                        ["seconds"] = Math.Round(seconds, 4),
                        ["rate"] = piece.SampleRate,
                        ["channels"] = piece.Channels,
                        ["source"] = path,
                        ["start"] = Math.Round((double)cut.Start / audio.SampleRate, 4),
                        ["end"] = Math.Round((double)cut.End / audio.SampleRate, 4),
                    };
                    if (text is not null && whole)
                    {
                        row["text"] = text;
                    }

                    manifest!.WriteLine(row.ToJsonString(CommandContext.JsonLine));
                }

                line.Advance();
            }

            line.Clear();
        }
        finally
        {
            manifest?.Dispose();
        }

        if (output is not null)
        {
            context.Write(string.Create(CultureInfo.InvariantCulture,
                $"wrote {pieces.Count:N0} file(s), {Total(pieces.Select(p => p.Seconds))} of sound, from {sources.Count:N0} source(s) ({Total(sources.Select(s => s.Seconds))}) into {output}; manifest: {Path.Combine(output, ManifestFile)}"));
            if (dropped > 0)
            {
                context.Write($"{dropped} source(s) gave no piece (all silence, or every piece under --min-seconds)");
            }

            if (unattached > 0)
            {
                context.Write($"{unattached} transcript(s) not attached: their files were split into pieces (a file's text is the whole file's)");
            }
        }

        if (stats)
        {
            var described = output is null ? sources.Select(s => (Name: s.Path, s.Seconds, s.Rate, s.Channels)).ToList() : pieces;
            WriteStats(context, output is null ? "inputs" : "written", described);
        }

        var json = new JsonObject
        {
            ["sources"] = sources.Count,
            ["source_seconds"] = Math.Round(sources.Sum(s => s.Seconds), 3),
            ["split"] = split,
            ["split_options"] = splitter is null ? null : splitOptions.ToJson(),
        };
        if (output is not null)
        {
            json["out"] = output;
            json["manifest"] = Path.Combine(output, ManifestFile);
            json["files"] = pieces.Count;
            json["seconds"] = Math.Round(pieces.Sum(p => p.Seconds), 3);
            json["rate"] = rate;
            json["mono"] = mono;
            json["bits"] = AudioFiles.BitsName(bits);
            json["transcripts_attached"] = attached;
            json["transcripts_unattached"] = unattached;
        }

        if (stats)
        {
            var described = output is null ? sources.Select(s => s.Seconds).ToList() : pieces.Select(p => p.Seconds).ToList();
            json["stats"] = StatsJson(described, output is null ? sources.Select(s => (s.Rate, s.Channels)) : pieces.Select(p => (p.Rate, p.Channels)));
        }

        if (context.Option("--align") is { } model && output is not null)
        {
            if (attached < pieces.Count)
            {
                throw new UsageException($"--align aligns pieces with their text: {pieces.Count - attached} of {pieces.Count} have none (give --transcripts with every file's text, "
                    + "and no --split: a split file's pieces have no text of their own).");
            }

            json["alignments"] = Align(context, model, new SpeechSource(SpeechSources.Transcripts, [Path.GetFullPath(Path.Combine(output, ManifestFile))]), output);
        }

        context.WriteJson(json);
        return ExitCodes.Ok;
    }

    // Whether --align reads the inputs as speech data (nothing converted): transcripts files, or folders and zips with no
    // conversion option asked for (a conversion writes the pieces first, then aligns them).
    private static bool AlignsOnly(CommandContext context)
    {
        bool transcripts = context.Positional.All(p => p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase));
        bool converts = context.Flag("--mono") || context.Flag("--stats")
            || new[] { "--rate", "--bits", "--type", "--split", "--clean", "--transcripts", "--min-seconds", "--max-seconds" }.Any(o => context.Option(o) is not null);
        bool folders = context.Positional.All(p => Directory.Exists(p) || p.EndsWith(".zip", StringComparison.OrdinalIgnoreCase));
        return transcripts || folders && !converts || context.Option("--source") is not null;
    }

    // Aligns `source` with the CTC model at `model` into DIR/alignments.jsonl: progress per batch, the notes, the count.
    private static string Align(CommandContext context, string model, SpeechSource source, string output)
    {
        using var recognizer = SpeechModels.Load(model, new SpeechModelOptions { Device = context.Device });
        var ctc = recognizer as ICtcSpeechModel
            ?? throw new UsageException($"--align {model} is a {recognizer.GetType().Name}: alignments come from a CTC speech model's best paths (an ICtcSpeechModel).");
        Directory.CreateDirectory(output);
        string path = Path.Combine(output, "alignments.jsonl");
        var options = new SpeechAlignmentOptions { BatchSeconds = AudioFiles.Number(context, "--align-seconds", new SpeechAlignmentOptions().BatchSeconds, double.Epsilon) };
        SpeechAlignmentResult result;
        using (var line = new ProgressLine(context, $"aligning with {Path.GetFileName(Path.TrimEndingDirectorySeparator(model))}", unit: ProgressUnit.Items))
        {
            try
            {
                result = SpeechAlignments.AlignToFile(ctc, source, path, options, new SyncProgress<SpeechAlignmentProgress>(p =>
                {
                    line.Total = p.Total;
                    line.Report(p.Done);
                }));
            }
            catch (Exception e) when (e is ArgumentException or InvalidDataException or FileNotFoundException)
            {
                line.Erase();
                throw new UsageException(e.Message);
            }

            line.Clear();
        }

        foreach (string note in result.Notes)
        {
            context.Write($"note: {note}");
        }

        context.Write(string.Create(CultureInfo.InvariantCulture, $"aligned {result.Alignments.Count:N0} utterance(s) in {result.Elapsed.TotalSeconds:F1} s: {path}"));
        return path;
    }

    // Each input file with the name it is written under: a folder's files under their path in the folder, others by name
    // (a second file of a name already taken gets -2, -3, ...).
    private static List<(string Path, string Relative)> Inputs(IEnumerable<string> arguments)
    {
        var inputs = new List<(string, string)>();
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (string argument in arguments)
        {
            bool folder = Directory.Exists(argument);
            foreach (string file in AudioFiles.Expand([argument]))
            {
                if (!seen.Add(Path.GetFullPath(file)))
                {
                    continue;
                }

                string relative = folder ? Path.GetRelativePath(argument, file) : Path.GetFileName(file);
                string unique = relative;
                for (int n = 2; !names.Add(Path.ChangeExtension(unique, null)); n++)
                {
                    unique = Path.Combine(Path.GetDirectoryName(relative) ?? "", $"{Path.GetFileNameWithoutExtension(relative)}-{n}{Path.GetExtension(relative)}");
                }

                inputs.Add((file, unique));
            }
        }

        return inputs;
    }

    // The texts by file name: JSON Lines rows, or NAME<tab>TEXT / NAME|TEXT lines.
    private static Dictionary<string, string> Transcripts(string path)
    {
        if (!File.Exists(path))
        {
            throw new UsageException($"Transcripts file not found: {path}");
        }

        var texts = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        int number = 0;
        foreach (string raw in File.ReadLines(path, Encoding.UTF8))
        {
            number++;
            string line = raw.Trim();
            if (line.Length == 0)
            {
                continue;
            }

            if (line[0] == '{')
            {
                var row = JsonNode.Parse(line) as JsonObject ?? throw new UsageException($"{path}:{number}: not a JSON object.");
                string? name = First(row, "file", "audio", "path", "name", "id");
                string? text = First(row, "text", "transcript", "sentence", "transcription");
                if (name is null || text is null)
                {
                    throw new UsageException($"{path}:{number}: a row needs \"file\" (or audio, path, name, id) and \"text\" (or transcript, sentence).");
                }

                texts[Path.GetFileName(name)] = text;
                continue;
            }

            int cut = line.IndexOfAny(['\t', '|']);
            if (cut <= 0)
            {
                throw new UsageException($"{path}:{number}: expected NAME<tab>TEXT or NAME|TEXT.");
            }

            texts[Path.GetFileName(line[..cut].Trim())] = line[(cut + 1)..].Trim();
        }

        return texts;
    }

    private static string? First(JsonObject row, params ReadOnlySpan<string> keys)
    {
        foreach (string key in keys)
        {
            if (row[key] is JsonValue value && value.TryGetValue(out string? text))
            {
                return text;
            }
        }

        return null;
    }

    // A file's text: by its name, then by its name without the extension (either side).
    private static string? Text(Dictionary<string, string> texts, string relative)
    {
        string name = Path.GetFileName(relative), stem = Path.GetFileNameWithoutExtension(relative);
        if (texts.TryGetValue(name, out var text) || texts.TryGetValue(stem, out text))
        {
            return text;
        }

        foreach (var (key, value) in texts)
        {
            if (string.Equals(Path.GetFileNameWithoutExtension(key), stem, StringComparison.OrdinalIgnoreCase))
            {
                return value;
            }
        }

        return null;
    }

    private static string Total(IEnumerable<double> seconds) => Units.Duration(TimeSpan.FromSeconds(seconds.Sum()));

    private static void WriteStats(CommandContext context, string what, IReadOnlyList<(string Name, double Seconds, int Rate, int Channels)> files)
    {
        if (files.Count == 0)
        {
            context.Write($"{what}: no files");
            return;
        }

        var lengths = files.Select(f => f.Seconds).Order().ToList();
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{what}: {files.Count:N0} file(s), {Total(lengths)}; length min {lengths[0]:0.00} s, median {lengths[lengths.Count / 2]:0.00} s, mean {lengths.Average():0.00} s, max {lengths[^1]:0.00} s"));
        context.Write("rates: " + string.Join(", ", files.GroupBy(f => f.Rate).OrderBy(g => g.Key).Select(g => string.Create(CultureInfo.InvariantCulture, $"{g.Key} Hz × {g.Count()}"))));
        context.Write("channels: " + string.Join(", ", files.GroupBy(f => f.Channels).OrderBy(g => g.Key).Select(g => $"{g.Key} × {g.Count()}")));
        var buckets = Histogram(lengths);
        int widest = buckets.Max(b => b.Count);
        char bar = context.Plain ? '#' : '█';
        foreach (var (from, to, count) in buckets)
        {
            int width = widest == 0 ? 0 : (int)Math.Round(40.0 * count / widest);
            context.Write(string.Create(CultureInfo.InvariantCulture, $"  {from,8:0.00} - {to,8:0.00} s  {count,6:N0}  {new string(bar, width)}"));
        }
    }

    // Ten buckets of equal width between the shortest and the longest length (one when they are equal).
    internal static List<(double From, double To, int Count)> Histogram(IReadOnlyList<double> sorted)
    {
        double min = sorted[0], max = sorted[^1];
        int buckets = max > min ? Math.Min(10, sorted.Count) : 1;
        double width = buckets == 1 ? Math.Max(max - min, 0) : (max - min) / buckets;
        var counts = new int[buckets];
        foreach (double v in sorted)
        {
            counts[buckets == 1 ? 0 : Math.Min(buckets - 1, (int)((v - min) / width))]++;
        }

        return [.. Enumerable.Range(0, buckets).Select(b => (min + b * width, buckets == 1 ? max : min + (b + 1) * width, counts[b]))];
    }

    private static JsonObject StatsJson(List<double> seconds, IEnumerable<(int Rate, int Channels)> formats)
    {
        var sorted = seconds.Order().ToList();
        var list = formats.ToList();
        return new JsonObject
        {
            ["files"] = sorted.Count,
            ["seconds"] = Math.Round(sorted.Sum(), 3),
            ["min_seconds"] = sorted.Count == 0 ? null : Math.Round(sorted[0], 4),
            ["max_seconds"] = sorted.Count == 0 ? null : Math.Round(sorted[^1], 4),
            ["mean_seconds"] = sorted.Count == 0 ? null : Math.Round(sorted.Average(), 4),
            ["rates"] = new JsonObject(list.GroupBy(f => f.Rate).OrderBy(g => g.Key).Select(g => KeyValuePair.Create(g.Key.ToString(CultureInfo.InvariantCulture), (JsonNode?)g.Count()))),
            ["channels"] = new JsonObject(list.GroupBy(f => f.Channels).OrderBy(g => g.Key).Select(g => KeyValuePair.Create(g.Key.ToString(CultureInfo.InvariantCulture), (JsonNode?)g.Count()))),
            ["histogram"] = sorted.Count == 0 ? new JsonArray() : new JsonArray([.. Histogram(sorted).Select(b => (JsonNode)new JsonObject
            {
                ["from"] = Math.Round(b.From, 4), ["to"] = Math.Round(b.To, 4), ["count"] = b.Count,
            })]),
        };
    }
}

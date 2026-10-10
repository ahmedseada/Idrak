// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Inference;
using Idrak.Inference.Abstractions;
using Idrak.Layers;
using Idrak.Vision;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// What a recognizer is: its alphabet, the line height it reads, its reading direction and its network's sizes. Saved
/// beside the network (<see cref="Recognizer.SettingsFile"/>) so reading prepares lines exactly as training did.
/// </summary>
internal sealed record RecognizerSettings
{
    /// <summary>The characters (Unicode scalars, NFC), class i + 1 each; class 0 is CTC's blank.</summary>
    public required IReadOnlyList<string> Alphabet { get; init; }

    /// <summary>The height every line is scaled to (a multiple of 4).</summary>
    public int Height { get; init; } = 48;

    /// <summary>Whether lines read right to left (each line image is flipped horizontally first; see <see cref="ReadingOrder"/>).</summary>
    public bool RightToLeft { get; init; } = true;

    /// <summary>The convolution blocks' channels (at least 3: the first two halve height and width, the rest the height).</summary>
    public IReadOnlyList<int> Channels { get; init; } = [32, 64, 128];

    /// <summary>The bidirectional LSTM's hidden size, each way.</summary>
    public int Hidden { get; init; } = 128;

    /// <summary>Stacked LSTM layers.</summary>
    public int Layers { get; init; } = 2;

    /// <summary>The augmentation pipeline training used (<see cref="Augmentations"/>), for the record.</summary>
    public string Augment { get; init; } = "";

    /// <summary>Classes the network scores: the alphabet and the blank.</summary>
    public int Classes => Alphabet.Count + 1;

    /// <summary>How many image columns make one step of the sequence (the two halving pools).</summary>
    public const int Stride = 4;

    public JsonObject ToJson() => new()
    {
        ["alphabet"] = new JsonArray([.. Alphabet.Select(a => (JsonNode)a)]),
        ["height"] = Height,
        ["direction"] = RightToLeft ? "rtl" : "ltr",
        ["channels"] = new JsonArray([.. Channels.Select(c => (JsonNode)c)]),
        ["hidden"] = Hidden,
        ["layers"] = Layers,
        ["augment"] = Augment,
        ["preprocessing"] = "grey, ink made light on dark (polarity from the line), contrast stretched to [0, 1], scaled to the height keeping the aspect, "
                            + "flipped horizontally for rtl, padded on the right with 0 to a multiple of 4 columns",
    };

    public static RecognizerSettings FromJson(JsonObject json) => new()
    {
        Alphabet = [.. (json["alphabet"] as JsonArray ?? throw new InvalidDataException("No \"alphabet\".")).Select(a => (string)a!)],
        Height = (int?)json["height"] ?? 48,
        RightToLeft = (string?)json["direction"] != "ltr",
        Channels = json["channels"] is JsonArray channels ? [.. channels.Select(c => (int)c!)] : [32, 64, 128],
        Hidden = (int?)json["hidden"] ?? 128,
        Layers = (int?)json["layers"] ?? 2,
        Augment = (string?)json["augment"] ?? "",
    };
}

/// <summary>A line prepared for the network: [height, width] pixels, ink high, in reading order (flipped for right to left).</summary>
internal sealed record PreparedLine(float[] Pixels, int Width)
{
    /// <summary>The network's steps for this line (its width over the stride, rounded up).</summary>
    public int Steps => (Width + RecognizerSettings.Stride - 1) / RecognizerSettings.Stride;

    /// <summary>
    /// <paramref name="line"/> as the recognizer reads it: grey; inverted when mostly light (ink on paper) so ink is high and
    /// the background 0 (padding, augmentation fills and the network all share that 0); stretched to [0, 1]; scaled to
    /// <paramref name="height"/> keeping its aspect; flipped horizontally for right to left.
    /// </summary>
    public static PreparedLine From(ImageData line, int height, bool rightToLeft)
    {
        var grey = Foreground.Grey(line);
        double mean = grey.Average();
        float lo = float.MaxValue, hi = float.MinValue;
        for (int i = 0; i < grey.Length; i++)
        {
            float v = mean > 0.5 ? 1f - grey[i] : grey[i];
            grey[i] = v;
            lo = Math.Min(lo, v);
            hi = Math.Max(hi, v);
        }

        float range = hi - lo;
        for (int i = 0; i < grey.Length; i++)
        {
            grey[i] = range > 1e-3f ? (grey[i] - lo) / range : 0f;
        }

        int width = Math.Max(RecognizerSettings.Stride, (int)Math.Round(line.Width * (double)height / line.Height));
        var pixels = new ImageData(grey, 1, line.Height, line.Width).Resize(1, height, width);
        if (rightToLeft)
        {
            for (int y = 0; y < height; y++)
            {
                Array.Reverse(pixels, y * width, width);
            }
        }

        return new PreparedLine(pixels, width);
    }
}

/// <summary>One line read: its text in logical order, the decoder's log-probability and a per-step confidence.</summary>
internal sealed record LineReading(string Text, double LogProbability, double Confidence);

/// <summary>
/// The line recognizer: a small convolutional network whose pooling ends at height 1 (2x2 twice, then 2x1, then an
/// average over what height is left), its columns read as a sequence by a bidirectional LSTM, scored over the alphabet and
/// CTC's blank. Built with the library's <see cref="NetworkBuilder"/>; saved as a package (<see cref="ModelPackage"/>)
/// with its settings beside it.
/// </summary>
internal sealed class Recognizer : IDisposable
{
    /// <summary>The network and its weights in the model folder.</summary>
    public const string PackageFile = "recognizer.ikm";

    /// <summary>The settings in the model folder (alphabet, height, direction, sizes, preprocessing).</summary>
    public const string SettingsFile = "recognizer.json";

    private Recognizer(RecognizerSettings settings, Sequential network, NetworkBuilder? builder, Device device)
    {
        Settings = settings;
        Network = network;
        Builder = builder;
        Device = device;
    }

    public RecognizerSettings Settings { get; }

    public Sequential Network { get; }

    public Device Device { get; }

    // The builder a new network came from (its description is saved with the weights).
    private NetworkBuilder? Builder { get; }

    /// <summary>The network's description for <paramref name="settings"/> (input [1, height, 64]; any width runs).</summary>
    public static NetworkBuilder Describe(RecognizerSettings settings, Device device, int seed)
    {
        if (settings.Height < 8 || settings.Height % 4 != 0)
        {
            throw new UsageException($"--height must be a multiple of 4, at least 8 (given {settings.Height}).");
        }

        if (settings.Channels.Count < 3)
        {
            throw new UsageException("--channels needs at least three blocks, such as 32,64,128.");
        }

        var builder = Layers.Network.Image(1, settings.Height, 64).OnDevice(device).Seed(seed).Named("line-recognizer");
        int height = settings.Height;
        for (int block = 0; block < settings.Channels.Count; block++)
        {
            builder = builder.Conv2d(settings.Channels[block], 3, padding: 1).BatchNorm().ReLU();
            if (block < 2)
            {
                builder = builder.MaxPool2d(2);
                height /= 2;
            }
            else if (height >= 4 && height % 2 == 0)
            {
                builder = builder.MaxPool2d((2, 1));
                height /= 2;
            }
        }

        if (height > 1)
        {
            builder = builder.AvgPool2d((height, 1));                                 // rectangular pooling to height 1
        }

        return builder.ColumnsToSequence().LSTM(settings.Hidden, returnSequences: true, bidirectional: true, layers: settings.Layers).Linear(settings.Classes);
    }

    /// <summary>A new recognizer with random weights.</summary>
    public static Recognizer Create(RecognizerSettings settings, Device device, int seed)
    {
        var builder = Describe(settings, device, seed);
        return new Recognizer(settings, builder.Build(), builder, device);
    }

    /// <summary>The recognizer saved in <paramref name="folder"/>.</summary>
    public static Recognizer Load(string folder, Device device)
    {
        string package = Path.Combine(folder, PackageFile), settingsFile = Path.Combine(folder, SettingsFile);
        if (!File.Exists(package) || !File.Exists(settingsFile))
        {
            throw new UsageException($"{folder} is not a recognizer folder (no {PackageFile} and {SettingsFile}); train one with: train --data DIR -o {folder}");
        }

        var settings = RecognizerSettings.FromJson((JsonObject)JsonNode.Parse(File.ReadAllText(settingsFile))!);
        using var reader = ModelPackage.Open(package);
        var network = reader.BuildNetwork(device: device);
        network.Eval();
        return new Recognizer(settings, network, null, device);
    }

    /// <summary>Writes the network, its weights and the settings (with <paramref name="summary"/>, the training's record) to <paramref name="folder"/>.</summary>
    public void Save(string folder, JsonObject? summary = null, string packageFile = PackageFile)
    {
        Directory.CreateDirectory(folder);
        var writer = ModelPackage.Create(Path.Combine(folder, packageFile));
        writer.Architecture(ModelPackage.DefaultModelName, Builder?.ToJson() ?? Layers.Network.ArchitectureOf(Network) ?? throw new InvalidOperationException("The network has no description to save."));
        writer.Weights(Network).Save();
        var json = Settings.ToJson();
        if (summary is not null)
        {
            json["training"] = summary;
        }

        OcrText.Write(Path.Combine(folder, SettingsFile), json.ToJsonString(Json.Indented));
    }

    /// <summary>The label ids of a transcription (logical order) in column order; characters outside the alphabet are left out and returned.</summary>
    public (int[] Labels, IReadOnlyList<string> Unknown) Encode(string transcription)
    {
        var index = Index().GetAlternateLookup<ReadOnlySpan<char>>();                  // looked up by the rune's chars: no string a character
        var labels = new List<int>();
        var unknown = new List<string>();
        Span<char> pair = stackalloc char[2];
        foreach (var rune in ReadingOrder.ToColumns(OcrText.Normalize(transcription)).EnumerateRunes())
        {
            var chars = pair[..rune.EncodeToUtf16(pair)];
            if (index.TryGetValue(chars, out int id))
            {
                labels.Add(id);
            }
            else
            {
                unknown.Add(chars.ToString());
            }
        }

        return ([.. labels], unknown);
    }

    /// <summary>The text of label ids in column order, back in logical order (NFC).</summary>
    public string Text(IEnumerable<int> labels)
    {
        var builder = new StringBuilder();
        foreach (int id in labels)
        {
            if (id >= 1 && id <= Settings.Alphabet.Count)
            {
                builder.Append(Settings.Alphabet[id - 1]);
            }
        }

        return OcrText.Normalize(ReadingOrder.FromColumns(builder.ToString()));
    }

    private Dictionary<string, int>? _index;

    private Dictionary<string, int> Index() => _index ??= Settings.Alphabet.Select((a, i) => (a, i)).ToDictionary(p => p.a, p => p.i + 1, StringComparer.Ordinal);

    /// <summary>A line as the network reads it.</summary>
    public PreparedLine Prepare(ImageData line) => PreparedLine.From(line, Settings.Height, Settings.RightToLeft);

    /// <summary>
    /// A batch of lines as one tensor [N, 1, height, widest] (each padded on the right with 0 to the widest, rounded up to
    /// the stride) and each line's steps.
    /// </summary>
    public (Tensor Images, int[] Steps) Batch(IReadOnlyList<PreparedLine> lines)
    {
        int height = Settings.Height;
        int width = lines.Max(l => l.Steps) * RecognizerSettings.Stride;
        var values = new float[lines.Count * height * width];
        for (int n = 0; n < lines.Count; n++)
        {
            var line = lines[n];
            for (int y = 0; y < height; y++)
            {
                Array.Copy(line.Pixels, y * line.Width, values, (n * height + y) * width, line.Width);
            }
        }

        return (Tensor.From(values, [lines.Count, 1, height, width], Device), [.. lines.Select(l => l.Steps)]);
    }

    /// <summary>The network's log-probabilities [N, steps, classes] for a batch.</summary>
    public Tensor LogProbabilities(Tensor images) => Network.Forward(images).LogSoftmax();

    /// <summary>
    /// Reads <paramref name="lines"/> with the CTC decoder <paramref name="decoder"/> (<see cref="CtcDecoders"/>), in
    /// batches measured on the device (<see cref="MeasuredBatches"/>), widest lines first so each batch pads little.
    /// </summary>
    public IReadOnlyList<LineReading> Read(IReadOnlyList<ImageData> lines, string decoder, CtcDecodeOptions options, MeasuredBatches batches)
    {
        var prepared = lines.Select(Prepare).ToArray();
        var order = Enumerable.Range(0, prepared.Length).OrderByDescending(i => prepared[i].Width).ToArray();
        var readings = new LineReading[prepared.Length];
        Network.Eval();
        batches.Run(order.Length, i => prepared[order[i]].Steps, (start, count) =>
        {
            using var scope = new TensorScope();
            var group = order.Skip(start).Take(count).Select(i => prepared[i]).ToArray();
            var (images, steps) = Batch(group);
            var logProbs = Network.Predict(images).LogSoftmax();
            var decoded = CtcDecoders.Decode(logProbs, steps, decoder, options with { Blank = 0 }, batchFirst: true);
            for (int k = 0; k < count; k++)
            {
                var best = decoded[k].Count > 0 ? decoded[k][0] : new CtcHypothesis([], 0f);
                readings[order[start + k]] = new LineReading(Text(best.Labels), best.LogProbability, Math.Exp(best.LogProbability / Math.Max(1, steps[k])));
            }
        });

        return readings;
    }

    public void Dispose() => Network.Dispose();
}

/// <summary>
/// How many lines go through the network at once, measured on the running device as the library's image commands do: the
/// first batch is one line (the widest), whose peak memory per step of width sets the rest (as many steps as fit in half
/// the memory the device reports free); a batch the device has no room for is halved and run again. A ceiling may be
/// given instead (<c>--batch</c>). A batch takes only items within an eighth of its first's width (items come widest
/// first), so little of it is padding. No card or memory size is assumed.
/// </summary>
internal sealed class MeasuredBatches(Device device, int? ceiling = null)
{
    private long? _perStep;
    private bool _measured;
    private int _halvedTo = int.MaxValue;

    /// <summary>Times a batch ran out of memory and was halved.</summary>
    public int Halvings { get; private set; }

    /// <summary>The largest batch run.</summary>
    public int Largest { get; private set; }

    /// <summary>Runs <paramref name="count"/> items (each <paramref name="steps"/> wide) through <paramref name="run"/> (start, length).</summary>
    public void Run(int count, Func<int, int> steps, Action<int, int> run)
    {
        for (int start = 0; start < count;)
        {
            int n = Next(start, count, steps);
            try
            {
                var backend = device.Backend;
                long before = backend.GetMemoryUsage().InUse;
                backend.ResetPeakMemoryUsage();
                run(start, n);
                if (!_measured)
                {
                    long used = Math.Max(1, backend.GetMemoryUsage().Peak - before);
                    _perStep = Math.Max(1, used / Math.Max(1, Enumerable.Range(start, n).Max(steps) * n));
                    _measured = true;
                }

                Largest = Math.Max(Largest, n);
                start += n;
            }
            catch (ResourceLimitExceededException) when (n > 1)
            {
                _halvedTo = n / 2;
                Halvings++;
            }
        }
    }

    // The next batch: one item until measured; then as many as fit (each padded to the first's width, the widest), of
    // those only the items within an eighth of the first's width (Similar).
    private int Next(int start, int count, Func<int, int> steps)
    {
        int left = count - start;
        int limit = Math.Min(left, Math.Min(_halvedTo, ceiling ?? int.MaxValue));
        if (!_measured)
        {
            return ceiling is null ? 1 : Similar(start, limit, steps);
        }

        if (_perStep is not { } perStep || device.Backend.AvailableMemory() is not { } free)
        {
            return Similar(start, limit, steps);
        }

        long budget = free / 2;
        long width = steps(start);
        return Similar(start, (int)Math.Clamp(budget / Math.Max(1, perStep * width), 1, limit), steps);
    }

    // Of `n` items from `start` (widest first), the leading ones at least seven eighths of the first's width. A line in a
    // batch is padded to the widest, and the recognizer's reverse LSTM reads that padding before the line: training's
    // batches (lines of similar width) hardly have any, so a reading with much more drifts; and padding is work for nothing.
    private static int Similar(int start, int n, Func<int, int> steps)
    {
        long first = steps(start);
        int similar = 1;
        while (similar < n && steps(start + similar) * 8L >= first * 7)
        {
            similar++;
        }

        return similar;
    }
}

/// <summary>JSON writing options of the app.</summary>
internal static class Json
{
    public static readonly System.Text.Json.JsonSerializerOptions Indented = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,                 // Arabic as itself, not \u escapes
    };
}

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision.Plugins.Crnn;

/// <summary>
/// What a CRNN line recognizer is: its alphabet, the line height it reads, its reading direction and order, and its
/// network's sizes. Saved in the model folder's config.json (<see cref="CrnnRecognizer.ConfigFile"/>, beside its
/// architecture name) so reading prepares lines exactly as training did.
/// </summary>
public sealed record CrnnSettings
{
    /// <summary>How many image columns make one step of the sequence (the two halving pools).</summary>
    public const int Stride = 4;

    /// <summary>The characters (Unicode scalars, NFC), class i + 1 each; class 0 is CTC's blank.</summary>
    public required IReadOnlyList<string> Alphabet { get; init; }

    /// <summary>The height every line is scaled to (a multiple of 4, at least 8).</summary>
    public int Height { get; init; } = 48;

    /// <summary>
    /// Whether lines read right to left (each line image is flipped horizontally first; the transcription is turned to the
    /// flipped line's column order by <see cref="ReadingOrder"/>).
    /// </summary>
    public bool RightToLeft { get; init; } = true;

    /// <summary>
    /// The reading order (a <see cref="ReadingOrders"/> name) that turns a transcription to the line's column order and a
    /// reading back: "runs" by default (digits and Latin runs reversed in a right-to-left line; UAX #9's "bidi" maps a
    /// number before a Latin word and after it to the same columns, so it cannot read them back).
    /// </summary>
    public string ReadingOrder { get; init; } = ReadingOrders.Runs;

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

    /// <summary>Each class's text in output order: the blank's empty, then the alphabet.</summary>
    public IReadOnlyList<string> Labels => ["", .. Alphabet];

    /// <summary>
    /// How the library's line reader (<see cref="ModelLineReader"/>) prepares a line for this network and turns its scores
    /// to text: <see cref="Height"/>, a step every <see cref="Stride"/> columns, the blank 0, the direction and reading
    /// order, ink made high, scores turned to log-probabilities by a log-softmax.
    /// </summary>
    public LineRecognitionSettings Lines => new() { Height = Height, Stride = Stride, Blank = 0, RightToLeft = RightToLeft, ReadingOrder = ReadingOrder };

    /// <summary>Why a network cannot be made from these settings, or null when it can.</summary>
    public string? Problem() =>
        Alphabet.Count == 0 ? "an alphabet"
        : Height < 8 || Height % 4 != 0 ? $"a line height that is a multiple of 4, at least 8 (given {Height})"
        : Channels.Count < 3 ? "at least three blocks of convolution channels, such as 32,64,128"
        : Channels.Any(c => c <= 0) || Hidden <= 0 || Layers <= 0 ? "positive channels, hidden size and layers"
        : null;

    /// <summary>The settings as config.json keys (with how lines are prepared, for the record).</summary>
    public JsonObject ToJson() => new()
    {
        ["alphabet"] = new JsonArray([.. Alphabet.Select(a => (JsonNode)a)]),
        ["height"] = Height,
        ["direction"] = RightToLeft ? "rtl" : "ltr",
        ["reading_order"] = ReadingOrder,
        ["channels"] = new JsonArray([.. Channels.Select(c => (JsonNode)c)]),
        ["hidden"] = Hidden,
        ["layers"] = Layers,
        ["augment"] = Augment,
        ["preprocessing"] = "grey, ink made light on dark (polarity from the line), contrast stretched to [0, 1], scaled to the height keeping the aspect, "
                            + "flipped horizontally for rtl, padded on the right with 0 to a multiple of 4 columns",
    };

    /// <summary>Settings from <see cref="ToJson"/>'s keys (absent ones take the defaults; "alphabet" is required).</summary>
    /// <exception cref="InvalidDataException">There is no alphabet, or the settings cannot make a network.</exception>
    public static CrnnSettings FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var settings = new CrnnSettings
        {
            Alphabet = [.. (json["alphabet"] as JsonArray ?? throw new InvalidDataException("A CRNN line recognizer's settings have no \"alphabet\".")).Select(a => (string)a!)],
            Height = (int?)json["height"] ?? 48,
            RightToLeft = (string?)json["direction"] != "ltr",
            ReadingOrder = (string?)json["reading_order"] ?? ReadingOrders.Runs,
            Channels = json["channels"] is JsonArray channels ? [.. channels.Select(c => (int)c!)] : [32, 64, 128],
            Hidden = (int?)json["hidden"] ?? 128,
            Layers = (int?)json["layers"] ?? 2,
            Augment = (string?)json["augment"] ?? "",
        };
        return settings.Problem() is { } problem ? throw new InvalidDataException($"A CRNN line recognizer's settings need {problem}.") : settings;
    }
}

/// <summary>
/// The CRNN line recognizer: a small convolutional network whose pooling ends at height 1 (2x2 twice, then 2x1, then an
/// average over what height is left), its columns read as a sequence by a bidirectional LSTM, scored over the alphabet and
/// CTC's blank. Built with the library's <see cref="NetworkBuilder"/>; lines are prepared, batched, decoded and turned to
/// text by the library's <see cref="ModelLineReader"/> (<see cref="ImageTask.LineRecognition"/>). Saved as a model folder
/// the family <see cref="CrnnPlugin.Architecture"/> reads (<see cref="ConfigFile"/> and <see cref="WeightsFile"/>), so
/// <c>ImageModels.Load</c>, <c>idrak predict</c> and <c>idrak train</c> read it too; a folder of the earlier layout
/// (<see cref="LegacyPackageFile"/> and <see cref="LegacySettingsFile"/>) still loads.
/// </summary>
public sealed class CrnnRecognizer : IDisposable
{
    /// <summary>The settings and the architecture name in the model folder.</summary>
    public const string ConfigFile = "config.json";

    /// <summary>The network's weights in the model folder (float32 safetensors, named by layer).</summary>
    public const string WeightsFile = "model.safetensors";

    /// <summary>The earlier layout's network and weights (an Idrak model package).</summary>
    public const string LegacyPackageFile = "recognizer.ikm";

    /// <summary>The earlier layout's settings.</summary>
    public const string LegacySettingsFile = "recognizer.json";

    private static readonly JsonSerializerOptions Indented = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,                       // every script as itself, not \u escapes
    };

    private readonly ModelLineReader _lines;
    private readonly Func<string, string>? _normalize;

    private CrnnRecognizer(CrnnSettings settings, Module network, Device device, Func<string, string>? normalize)
    {
        Settings = settings;
        Network = network;
        Device = device;
        _normalize = normalize;
        _lines = new ModelLineReader(network, settings.Labels, settings.Lines, device: device);
    }

    /// <summary>What the recognizer is.</summary>
    public CrnnSettings Settings { get; }

    /// <summary>The network: lines [N, 1, height, width] in, [N, width / stride, classes] scores out.</summary>
    public Module Network { get; }

    /// <summary>Where the network runs.</summary>
    public Device Device { get; }

    /// <summary>The network's description for <paramref name="settings"/> (input [1, height, 64]; any width runs).</summary>
    /// <exception cref="ArgumentException">The settings cannot make a network (<see cref="CrnnSettings.Problem"/>).</exception>
    public static NetworkBuilder Describe(CrnnSettings settings, Device device, int seed)
    {
        ArgumentNullException.ThrowIfNull(settings);
        if (settings.Problem() is { } problem)
        {
            throw new ArgumentException($"A CRNN line recognizer needs {problem}.", nameof(settings));
        }

        var builder = Idrak.Layers.Network.Image(1, settings.Height, 64).OnDevice(device).Seed(seed).Named("line-recognizer");
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
    /// <param name="settings">What it is.</param>
    /// <param name="device">Where the network is made.</param>
    /// <param name="seed">The weights' seed.</param>
    /// <param name="textNormalizer">
    /// Applied to a transcription before it is encoded and to every reading (an app's own form of text, such as NFC with
    /// white space collapsed); null keeps text as it is.
    /// </param>
    public static CrnnRecognizer Create(CrnnSettings settings, Device device, int seed, Func<string, string>? textNormalizer = null) =>
        new(settings, Describe(settings, device, seed).Build(), device, textNormalizer);

    /// <summary>Whether <paramref name="folder"/> holds a recognizer (this layout's or the earlier one).</summary>
    public static bool IsRecognizerFolder(string folder) =>
        File.Exists(Path.Combine(folder, ConfigFile)) && File.Exists(Path.Combine(folder, WeightsFile))
        || File.Exists(Path.Combine(folder, LegacyPackageFile)) && File.Exists(Path.Combine(folder, LegacySettingsFile));

    /// <summary>
    /// The recognizer saved in <paramref name="folder"/>: read through its family (<see cref="ImageModels.Load"/>, so the
    /// family registered under <see cref="CrnnPlugin.Architecture"/> reads it: <see cref="CrnnPlugin.Register"/>), or, for
    /// a folder of the earlier layout, from its model package.
    /// </summary>
    /// <param name="folder">The model folder.</param>
    /// <param name="device">Where the network is made.</param>
    /// <param name="textNormalizer">As <see cref="Create"/>'s.</param>
    /// <exception cref="FileNotFoundException">The folder holds no recognizer.</exception>
    /// <exception cref="NotSupportedException">The family is not registered.</exception>
    public static CrnnRecognizer Load(string folder, Device device, Func<string, string>? textNormalizer = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        ArgumentNullException.ThrowIfNull(device);
        if (File.Exists(Path.Combine(folder, ConfigFile)))
        {
            var model = ImageModels.Load(folder, new ImageModelOptions { Device = device });
            try
            {
                var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, ConfigFile))) as JsonObject ?? throw new InvalidDataException($"{folder}: config.json is not a JSON object.");
                return new CrnnRecognizer(CrnnSettings.FromJson(config), model.Network, device, textNormalizer);
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        string package = Path.Combine(folder, LegacyPackageFile), settingsFile = Path.Combine(folder, LegacySettingsFile);
        if (!File.Exists(package) || !File.Exists(settingsFile))
        {
            throw new FileNotFoundException($"{folder} is not a line recognizer folder (no {ConfigFile} and {WeightsFile}, nor {LegacyPackageFile} and {LegacySettingsFile}).");
        }

        var settings = CrnnSettings.FromJson(JsonNode.Parse(File.ReadAllText(settingsFile)) as JsonObject ?? throw new InvalidDataException($"{settingsFile} is not a JSON object."));
        using var reader = ModelPackage.Open(package);
        var network = reader.BuildNetwork(device: device);
        network.Eval();
        return new CrnnRecognizer(settings, network, device, textNormalizer);
    }

    /// <summary>
    /// Writes the recognizer to <paramref name="folder"/>: config.json (the architecture name, the settings and
    /// <paramref name="summary"/>, the training's record, under "training") and the weights.
    /// </summary>
    public void Save(string folder, JsonObject? summary = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        Directory.CreateDirectory(folder);
        SafeTensorsWriter.Write(Path.Combine(folder, WeightsFile), [.. CrnnFamily.Tensors(Network).Select(t => (t.Name, t.Tensor.Shape.ToArray(), t.Tensor.ToArray()))],
            SafeTensorType.F32);
        var json = new JsonObject { ["architectures"] = new JsonArray(CrnnPlugin.Architecture) };
        foreach (var (key, value) in Settings.ToJson())
        {
            json[key] = value?.DeepClone();
        }

        if (summary is not null)
        {
            json["training"] = summary.DeepClone();
        }

        WriteConfig(folder, json);
    }

    /// <summary>Writes <paramref name="config"/> as the folder's config.json (UTF-8 without a BOM, indented, every script as itself).</summary>
    public static void WriteConfig(string folder, JsonObject config)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        ArgumentNullException.ThrowIfNull(config);
        File.WriteAllText(Path.Combine(folder, ConfigFile), config.ToJsonString(Indented).ReplaceLineEndings(Environment.NewLine) + Environment.NewLine,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    /// <summary>The label ids of a transcription (logical order) in column order; characters outside the alphabet are left out and returned.</summary>
    public (int[] Labels, IReadOnlyList<string> Unknown) Encode(string transcription) => _lines.Encode(Normalized(transcription));

    /// <summary>The text of label ids in column order, back in logical order.</summary>
    public string Text(IEnumerable<int> labels) => Normalized(_lines.Text(labels));

    /// <summary>A line as the network reads it (<see cref="ModelLineReader.Prepare(ImageData, LineRecognitionSettings)"/> with <see cref="CrnnSettings.Lines"/>).</summary>
    public PreparedLine Prepare(ImageData line) => _lines.Prepare(line);

    /// <summary>
    /// A batch of lines as one tensor [N, 1, height, widest] (each padded on the right with 0 to the widest, rounded up to
    /// the stride) and each line's steps.
    /// </summary>
    public (Tensor Images, int[] Steps) Batch(IReadOnlyList<PreparedLine> lines) => _lines.Batch(lines);

    /// <summary>The network's log-probabilities [N, steps, classes] for a batch, recording gradients.</summary>
    public Tensor LogProbabilities(Tensor images) => Network.Forward(images).LogSoftmax();

    /// <summary>
    /// Reads <paramref name="lines"/> with the CTC decoder <paramref name="decoder"/> (<see cref="CtcDecoders"/>), in
    /// batches measured on the device (<paramref name="batches"/>), widest lines first so each batch pads little.
    /// </summary>
    public IReadOnlyList<RecognizedLine> Read(IReadOnlyList<ImageData> lines, string decoder, CtcDecodeOptions options, MeasuredLineBatches batches)
    {
        ArgumentNullException.ThrowIfNull(lines);
        ArgumentNullException.ThrowIfNull(batches);
        var prepared = lines.Select(Prepare).ToArray();
        var order = Enumerable.Range(0, prepared.Length).OrderByDescending(i => prepared[i].Width).ToArray();
        var readings = new RecognizedLine[prepared.Length];
        var reader = new ModelLineReader(Network, Settings.Labels, Settings.Lines, decoder: decoder, options: options, device: Device);
        Network.Eval();
        batches.Run(order.Length, i => prepared[order[i]].Steps(CrnnSettings.Stride), (start, count) =>
        {
            var read = reader.Read([.. order.Skip(start).Take(count).Select(i => prepared[i])]);
            for (int k = 0; k < count; k++)
            {
                readings[order[start + k]] = read[k] with { Text = Normalized(read[k].Text) };
            }
        });

        return readings;
    }

    /// <summary>Disposes the network.</summary>
    public void Dispose() => Network.Dispose();

    private string Normalized(string text) => _normalize is null ? text : _normalize(text);
}

/// <summary>
/// How many lines go through a line recognizer at once, measured on the running device as the library's image commands do:
/// the first batch is one line (the widest), whose peak memory per step of width sets the rest (as many steps as fit in
/// half the memory the device reports free); a batch the device has no room for is halved and run again. A ceiling may be
/// given instead. A batch takes only items within an eighth of its first's width (items come widest first), so little of
/// it is padding. No card or memory size is assumed.
/// </summary>
/// <param name="device">The device the batches run on.</param>
/// <param name="ceiling">The most lines a batch takes (null: as many as the measured memory allows).</param>
public sealed class MeasuredLineBatches(Device device, int? ceiling = null)
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
        ArgumentNullException.ThrowIfNull(steps);
        ArgumentNullException.ThrowIfNull(run);
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

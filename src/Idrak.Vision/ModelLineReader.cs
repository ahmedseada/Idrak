// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>A line ready for a line recognizer: [<see cref="Height"/>, <see cref="Width"/>] grey values, in reading order.</summary>
/// <param name="Pixels">The values, row by row.</param>
/// <param name="Height">The recognizer's line height.</param>
/// <param name="Width">The line's width at that height (at least the stride).</param>
public sealed record PreparedLine(float[] Pixels, int Height, int Width)
{
    /// <summary>The network's steps for this line: its width over <paramref name="stride"/>, rounded up.</summary>
    public int Steps(int stride) => (Width + stride - 1) / stride;
}

/// <summary>One line read: its text (logical order), its labels (column order), the decoder's log-probability and a per-step confidence.</summary>
/// <param name="Text">The text, in logical order (the model's reading order applied).</param>
/// <param name="Labels">The decoded class ids, in column order, blanks and repeats gone.</param>
/// <param name="LogProbability">The decoder's score of the reading.</param>
/// <param name="Confidence">exp(log-probability / steps): the reading's probability per step, 0 to 1.</param>
public sealed record RecognizedLine(string Text, int[] Labels, double LogProbability, double Confidence);

/// <summary>
/// A line recognizer over a network (<see cref="ImageTask.LineRecognition"/>; <c>model.LineReader()</c> for a model loaded
/// through its family): each line prepared as <see cref="LineRecognitionSettings"/> say (scaled to the height keeping its
/// aspect, ink high, flipped for right to left), a batch padded with 0 on the right to its widest line, the network's
/// [N, steps, classes] scores decoded by a CTC decoder (<see cref="CtcDecoders"/>), the labels' text turned to logical
/// order by the model's reading order (<see cref="ReadingOrders"/>). It uses the network without owning it. A line in a
/// batch is padded to the widest; a network that reads its padding (a recurrent layer running backwards) reads lines of
/// similar widths best together, so callers batch lines by width.
/// </summary>
public sealed class ModelLineReader
{
    private readonly Module _network;
    private readonly Dictionary<string, int> _index;
    private readonly IReadingOrder? _order;

    /// <summary>A reader over <paramref name="network"/>.</summary>
    /// <param name="network">Images [N, <paramref name="channels"/>, height, width] in, [N, steps, classes] scores out.</param>
    /// <param name="labels">Each class's text, in output order (the blank's empty).</param>
    /// <param name="settings">How lines are prepared and readings turned to text.</param>
    /// <param name="channels">The network's input channels (the grey line repeated to them).</param>
    /// <param name="decoder">The CTC decoder (<see cref="CtcDecoders"/>; "greedy" unless given).</param>
    /// <param name="options">The decoder's settings (its blank is the settings').</param>
    /// <param name="device">Where the network runs (its weights' device unless given).</param>
    /// <exception cref="ArgumentException">The settings do not fit the labels.</exception>
    /// <exception cref="NotSupportedException">The decoder or the reading order is not registered.</exception>
    public ModelLineReader(Module network, IReadOnlyList<string> labels, LineRecognitionSettings settings, int channels = 1, string decoder = CtcDecoders.Greedy,
        CtcDecodeOptions? options = null, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        if (settings.Problem(labels.Count) is { } problem)
        {
            throw new ArgumentException($"Line recognition settings with {labels.Count} labels: {problem}.", nameof(settings));
        }

        _ = CtcDecoders.Get(decoder);                                              // refused here, not at the first line
        _network = network;
        Labels = labels;
        Settings = settings;
        Channels = channels;
        Decoder = decoder;
        Options = (options ?? new CtcDecodeOptions()) with { Blank = settings.Blank };
        Device = device ?? network.WeightsDevice ?? Device.Default;
        _order = settings.ReadingOrder is { } order ? ReadingOrders.Get(order) : null;
        _index = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < labels.Count; i++)
        {
            if (i != settings.Blank && labels[i].Length > 0)
            {
                _index.TryAdd(labels[i], i);
            }
        }
    }

    /// <summary>Each class's text, in output order.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>How lines are prepared and readings turned to text.</summary>
    public LineRecognitionSettings Settings { get; }

    /// <summary>The network's input channels.</summary>
    public int Channels { get; }

    /// <summary>The CTC decoder's name.</summary>
    public string Decoder { get; }

    /// <summary>The decoder's settings.</summary>
    public CtcDecodeOptions Options { get; }

    /// <summary>Where the network runs.</summary>
    public Device Device { get; }

    /// <summary>
    /// <paramref name="line"/> as a recognizer of <paramref name="settings"/> reads it: grey; with
    /// <see cref="LineRecognitionSettings.InkHigh"/>, inverted when mostly light (ink on paper) so ink is high and the
    /// background 0 (padding, augmentation fills and the network all share that 0) and stretched to [0, 1]; scaled to the
    /// height keeping its aspect (at least the stride wide); flipped horizontally for right to left.
    /// </summary>
    public static PreparedLine Prepare(ImageData line, LineRecognitionSettings settings)
    {
        ArgumentNullException.ThrowIfNull(line);
        ArgumentNullException.ThrowIfNull(settings);
        int height = settings.Height;
        var grey = Foreground.Grey(line);
        if (settings.InkHigh)
        {
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
        }

        int width = Math.Max(settings.Stride, (int)Math.Round(line.Width * (double)height / line.Height));
        var pixels = new ImageData(grey, 1, line.Height, line.Width).Resize(1, height, width);
        if (settings.RightToLeft)
        {
            for (int y = 0; y < height; y++)
            {
                Array.Reverse(pixels, y * width, width);
            }
        }

        return new PreparedLine(pixels, height, width);
    }

    /// <summary><paramref name="line"/> as this reader's network reads it (<see cref="Prepare(ImageData, LineRecognitionSettings)"/>).</summary>
    public PreparedLine Prepare(ImageData line) => Prepare(line, Settings);

    /// <summary>
    /// A batch of lines as one tensor [N, channels, height, widest] (each padded on the right with 0 to the widest, rounded
    /// up to the stride; the grey repeated to the channels) and each line's steps.
    /// </summary>
    public (Tensor Images, int[] Steps) Batch(IReadOnlyList<PreparedLine> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);
        if (lines.Count == 0)
        {
            throw new ArgumentException("A batch needs at least one line.", nameof(lines));
        }

        int height = Settings.Height, stride = Settings.Stride;
        int width = lines.Max(l => l.Steps(stride)) * stride;
        var values = new float[(long)lines.Count * Channels * height * width];
        var steps = new int[lines.Count];
        for (int n = 0; n < lines.Count; n++)
        {
            var line = lines[n];
            if (line.Height != height)
            {
                throw new ArgumentException($"Line {n + 1} is {line.Height} high; the recognizer reads lines {height} high.", nameof(lines));
            }

            steps[n] = line.Steps(stride);
            for (int c = 0; c < Channels; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    Array.Copy(line.Pixels, y * line.Width, values, ((long)(n * Channels + c) * height + y) * width, line.Width);
                }
            }
        }

        return (Tensor.From(values, [lines.Count, Channels, height, width], Device), steps);
    }

    /// <summary>The network's scores as log-probabilities over the classes (a log-softmax unless they are already).</summary>
    public Tensor LogProbabilities(Tensor scores)
    {
        ArgumentNullException.ThrowIfNull(scores);
        return Settings.LogProbabilities ? scores : scores.LogSoftmax();
    }

    /// <summary>Reads <paramref name="lines"/> as one batch (prepared in parallel).</summary>
    public IReadOnlyList<RecognizedLine> Read(IReadOnlyList<ImageData> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);
        var prepared = new PreparedLine[lines.Count];
        if (lines.Count > 1)
        {
            Parallel.For(0, lines.Count, ComputeResources.ParallelOptions, i => prepared[i] = Prepare(lines[i]));
        }
        else if (lines.Count == 1)
        {
            prepared[0] = Prepare(lines[0]);
        }

        return Read(prepared);
    }

    /// <summary>Reads prepared <paramref name="lines"/> as one batch.</summary>
    public IReadOnlyList<RecognizedLine> Read(IReadOnlyList<PreparedLine> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);
        if (lines.Count == 0)
        {
            return [];
        }

        using var scope = new TensorScope();
        var (images, steps) = Batch(lines);
        var scores = _network.Predict(images);
        if (scores.Rank != 3 || scores.Shape[0] != lines.Count || scores.Shape[2] != Labels.Count)
        {
            throw new InvalidOperationException($"The line recognizer gave scores [{string.Join(", ", scores.Shape.ToArray())}]; it gives [{lines.Count}, steps, {Labels.Count}] (a line, its steps, a score per class).");
        }

        int available = scores.Shape[1];
        for (int n = 0; n < steps.Length; n++)
        {
            steps[n] = Math.Min(steps[n], available);
        }

        var decoded = CtcDecoders.Decode(LogProbabilities(scores), steps, Decoder, Options, batchFirst: true);
        var result = new RecognizedLine[lines.Count];
        for (int n = 0; n < lines.Count; n++)
        {
            var best = decoded[n].Count > 0 ? decoded[n][0] : new CtcHypothesis([], 0f);
            result[n] = new RecognizedLine(Text(best.Labels), best.Labels, best.LogProbability, Math.Exp(best.LogProbability / Math.Max(1, steps[n])));
        }

        return result;
    }

    /// <summary>The text of class ids in column order, in logical order (the reading order applied); the blank and ids out of range add nothing.</summary>
    public string Text(IEnumerable<int> labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        var builder = new StringBuilder();
        foreach (int id in labels)
        {
            if (id >= 0 && id < Labels.Count && id != Settings.Blank)
            {
                builder.Append(Labels[id]);
            }
        }

        string columns = builder.ToString();
        return _order is null ? columns : _order.FromColumns(columns, Settings.RightToLeft);
    }

    /// <summary>
    /// The class ids of a transcription (logical order) in column order, a class per Unicode scalar; characters no label
    /// holds are left out and returned.
    /// </summary>
    public (int[] Labels, IReadOnlyList<string> Unknown) Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string columns = _order is null ? text : _order.ToColumns(text, Settings.RightToLeft);
        var index = _index.GetAlternateLookup<ReadOnlySpan<char>>();               // looked up by the scalar's chars: no string a character
        var labels = new List<int>(columns.Length);
        var unknown = new List<string>();
        Span<char> pair = stackalloc char[2];
        foreach (var rune in columns.EnumerateRunes())
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
}

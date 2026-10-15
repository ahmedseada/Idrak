// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace Idrak.Audio;

/// <summary>
/// Where each label of an utterance is spoken, from a CTC model's best path (<see cref="CtcAlignment.Viterbi(Tensor, ReadOnlySpan{int}, int)"/>):
/// the durations a synthesizer trains on (plan 21, S6). One row of an alignments file (<see cref="SpeechAlignments"/>).
/// </summary>
/// <param name="Audio">The utterance's audio, as its source gives it (<see cref="SpeechUtterance.Audio"/>).</param>
/// <param name="Text">What is said.</param>
/// <param name="Labels">The CTC model's labels of the text, in order.</param>
/// <param name="Starts">Each label's first step.</param>
/// <param name="Durations">Each label's steps on the path (the blanks around it not counted).</param>
/// <param name="Steps">The utterance's steps (the model's output steps for its frames).</param>
/// <param name="StepSeconds">The seconds a step spans (the hop times the model's subsampling).</param>
/// <param name="LogProbability">The path's log-probability.</param>
public sealed record SpeechAlignment(string Audio, string Text, int[] Labels, int[] Starts, int[] Durations, int Steps, double StepSeconds, float LogProbability)
{
    /// <summary>
    /// Each label's frames for a spectrogram of <paramref name="frames"/> frames of the same sound, summing to it: label i
    /// covers the steps from its start to the next label's (the first from step 0, the last to the end), so the blanks
    /// (silence) go to the label before them; the step boundaries scaled to frames and rounded.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="frames"/> is negative.</exception>
    public int[] FrameDurations(int frames)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(frames);
        int count = Labels.Length;
        var durations = new int[count];
        if (count == 0)
        {
            return durations;
        }

        double scale = Steps > 0 ? frames / (double)Steps : 0;
        int previous = 0;
        for (int i = 0; i < count; i++)
        {
            int boundary = i + 1 == count ? frames : Math.Clamp((int)Math.Round(Starts[i + 1] * scale, MidpointRounding.AwayFromZero), previous, frames);
            durations[i] = boundary - previous;
            previous = boundary;
        }

        return durations;
    }
}

/// <summary>How <see cref="SpeechAlignments.Align"/> runs. Every value is a setting.</summary>
public sealed record SpeechAlignmentOptions
{
    /// <summary>Audio seconds in one pass of the model, padding included (utterances sorted by length, so the padding is small).</summary>
    public double BatchSeconds { get; init; } = 60;

    /// <summary>Threads that read, decode and take the features of a batch (0: the library's thread budget).</summary>
    public int Workers { get; init; }
}

/// <summary>Where an alignment run is (<see cref="SpeechAlignments.Align"/>'s progress), after each batch.</summary>
/// <param name="Done">Utterances aligned (or left out).</param>
/// <param name="Total">Utterances in all.</param>
/// <param name="Elapsed">Since the run began.</param>
/// <param name="Remaining">The time left at the run's pace.</param>
/// <param name="AudioSecondsPerSecond">Seconds of audio aligned a second.</param>
public sealed record SpeechAlignmentProgress(int Done, int Total, TimeSpan Elapsed, TimeSpan Remaining, double AudioSecondsPerSecond);

/// <summary>What an alignment run did.</summary>
/// <param name="Alignments">One per utterance aligned, in the source's order.</param>
/// <param name="Notes">What was left out and why, one line each (the source's notes first).</param>
/// <param name="Elapsed">The time it took.</param>
public sealed record SpeechAlignmentResult(IReadOnlyList<SpeechAlignment> Alignments, IReadOnlyList<string> Notes, TimeSpan Elapsed);

/// <summary>
/// Alignments of transcribed speech: a CTC model's best path through each utterance's labels
/// (<see cref="Align"/>, what <c>idrak data audio --align MODEL</c> runs), and the alignments file a synthesizer trains
/// on (<see cref="SpeechTrainingOptions.Durations"/>). The file is JSON lines, one object per utterance:
/// <c>{"audio": path, "text": text, "labels": [ids], "starts": [steps], "durations": [steps], "steps": n,
/// "step_seconds": s, "log_probability": p}</c>, UTF-8, read from its bytes (rule 74); unknown keys are skipped.
/// </summary>
public static class SpeechAlignments
{
    /// <summary>
    /// Aligns every utterance of <paramref name="source"/> with <paramref name="model"/> (in evaluation mode, gradients
    /// off): features taken on worker threads, utterances batched by length (<see cref="SpeechAlignmentOptions.BatchSeconds"/>),
    /// one pass of the model per batch, each utterance's Viterbi path on the host. An utterance whose steps cannot hold
    /// its labels, or whose text has no label, is left out with a note.
    /// </summary>
    /// <exception cref="ArgumentException">An option is out of range.</exception>
    public static SpeechAlignmentResult Align(ICtcSpeechModel model, SpeechSource source, SpeechAlignmentOptions? options = null,
        IProgress<SpeechAlignmentProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(source);
        options ??= new SpeechAlignmentOptions();
        if (!(options.BatchSeconds > 0) || options.Workers < 0)
        {
            throw new ArgumentException("Speech alignment: BatchSeconds is positive and Workers not negative.", nameof(options));
        }

        var clock = Stopwatch.StartNew();
        using var reader = SpeechSources.Open(source);
        var notes = new List<string>(reader.Notes);
        var extractor = model.Features;
        int mels = extractor.Options.Mels, rate = extractor.Options.SampleRate;
        double secondsPerFrame = extractor.Options.HopSize / (double)rate;
        var order = Enumerable.Range(0, reader.Count).OrderBy(i => reader[i].Seconds).ThenBy(i => i).ToArray();
        var found = new SpeechAlignment?[reader.Count];
        int tooShort = 0, empty = 0, done = 0;
        double aligned = 0;
        var parallel = options.Workers > 0 ? new ParallelOptions { MaxDegreeOfParallelism = options.Workers } : ComputeResources.ParallelOptions;
        bool training = model.Network.IsTraining;
        model.Network.Eval();
        try
        {
            int start = 0;
            while (start < order.Length)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int end = start + 1;
                while (end < order.Length && (end - start + 1) * reader[order[end]].Seconds <= options.BatchSeconds)
                {
                    end++;
                }

                var batch = order[start..end];
                var features = new float[batch.Length][];
                var frames = new int[batch.Length];
                Parallel.For(0, batch.Length, parallel, i => (features[i], frames[i]) = Features(reader, batch[i], extractor));
                int longest = frames.Max();
                var padded = new float[batch.Length * longest * mels];
                for (int i = 0; i < batch.Length; i++)
                {
                    features[i].AsSpan(0, frames[i] * mels).CopyTo(padded.AsSpan(i * longest * mels));
                }

                float[] logProbs;
                int steps;
                using (var scope = new TensorScope())
                using (Autograd.NoGrad())
                {
                    var output = model.Logits(Tensor.From(padded, [batch.Length, longest, mels], model.Device), frames).LogSoftmax();
                    steps = output.Shape[1];
                    logProbs = output.ToArray();
                }

                int classes = model.Classes;
                for (int i = 0; i < batch.Length; i++)
                {
                    var utterance = reader[batch[i]];
                    int[] labels = model.Labels(utterance.Text);
                    int own = Math.Min(steps, model.StepsFor(frames[i]));
                    if (labels.Length == 0)
                    {
                        empty++;
                        continue;
                    }

                    if (own < CtcAlignment.MinimumSteps(labels))
                    {
                        tooShort++;
                        continue;
                    }

                    var path = CtcAlignment.Viterbi(logProbs.AsSpan(i * steps * classes, own * classes), own, classes, labels, model.Blank);
                    found[batch[i]] = new SpeechAlignment(utterance.Audio, utterance.Text, labels, path.Starts, path.Lengths, own, frames[i] * secondsPerFrame / own, path.LogProbability);
                    aligned += frames[i] * secondsPerFrame;
                }

                done += batch.Length;
                start = end;
                if (progress is not null)
                {
                    var elapsed = clock.Elapsed;
                    progress.Report(new SpeechAlignmentProgress(done, order.Length, elapsed, TimeSpan.FromSeconds(elapsed.TotalSeconds / done * (order.Length - done)),
                        aligned / Math.Max(1e-9, elapsed.TotalSeconds)));
                }
            }
        }
        finally
        {
            model.Network.Train(training);
        }

        if (tooShort > 0)
        {
            notes.Add($"{tooShort} utterances give fewer steps than their labels need (CTC cannot align them) and are left out");
        }

        if (empty > 0)
        {
            notes.Add($"{empty} utterances have no label (an empty text) and are left out");
        }

        return new SpeechAlignmentResult([.. found.OfType<SpeechAlignment>()], notes, clock.Elapsed);
    }

    /// <summary><see cref="Align"/>, the alignments written to <paramref name="path"/> (<see cref="Write"/>).</summary>
    public static SpeechAlignmentResult AlignToFile(ICtcSpeechModel model, SpeechSource source, string path, SpeechAlignmentOptions? options = null,
        IProgress<SpeechAlignmentProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var result = Align(model, source, options, progress, cancellationToken);
        Write(path, result.Alignments);
        return result;
    }

    /// <summary>Writes <paramref name="alignments"/> as an alignments file (JSON lines, UTF-8, one row per alignment).</summary>
    public static void Write(string path, IEnumerable<SpeechAlignment> alignments)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(alignments);
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { Length: > 0 } folder)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
        foreach (var a in alignments)
        {
            ArgumentNullException.ThrowIfNull(a);
            writer.WriteStartObject();
            writer.WriteString("audio"u8, a.Audio);
            writer.WriteString("text"u8, a.Text);
            Numbers(writer, "labels"u8, a.Labels);
            Numbers(writer, "starts"u8, a.Starts);
            Numbers(writer, "durations"u8, a.Durations);
            writer.WriteNumber("steps"u8, a.Steps);
            writer.WriteNumber("step_seconds"u8, a.StepSeconds);
            writer.WriteNumber("log_probability"u8, float.IsFinite(a.LogProbability) ? a.LogProbability : float.MinValue);
            writer.WriteEndObject();
            writer.Flush();
            stream.WriteByte((byte)'\n');
            writer.Reset();
        }
    }

    private static void Numbers(Utf8JsonWriter writer, ReadOnlySpan<byte> name, int[] values)
    {
        writer.WriteStartArray(name);
        foreach (int v in values)
        {
            writer.WriteNumberValue(v);
        }

        writer.WriteEndArray();
    }

    /// <summary>Reads an alignments file <see cref="Write"/> (or <see cref="AlignToFile"/>) wrote.</summary>
    /// <exception cref="FileNotFoundException">There is no such file.</exception>
    /// <exception cref="InvalidDataException">A row is not an alignment (naming the file and line).</exception>
    public static IReadOnlyList<SpeechAlignment> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"The alignments file {path} was not found (write one with SpeechAlignments.AlignToFile, or idrak data audio --align MODEL).", path);
        }

        var data = File.ReadAllBytes(path).AsSpan();
        if (data.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
        {
            data = data[3..];
        }

        var rows = new List<SpeechAlignment>();
        int line = 0;
        while (data.Length > 0)
        {
            line++;
            int end = data.IndexOf((byte)'\n');
            var row = (end < 0 ? data : data[..end]).TrimEnd((byte)'\r');
            data = end < 0 ? default : data[(end + 1)..];
            if (!row.Trim((byte)' ').IsEmpty)
            {
                rows.Add(Row(row, path, line));
            }
        }

        return rows;
    }

    private static SpeechAlignment Row(ReadOnlySpan<byte> row, string path, int line)
    {
        string? audio = null, text = null;
        int[]? labels = null, starts = null, durations = null;
        int steps = -1;
        double stepSeconds = 0;
        float logProbability = 0;
        try
        {
            var json = new Utf8JsonReader(row);
            if (!json.Read() || json.TokenType != JsonTokenType.StartObject)
            {
                throw new InvalidDataException("an alignment row is a JSON object");
            }

            while (json.Read() && json.TokenType == JsonTokenType.PropertyName)
            {
                var key = json.ValueSpan;
                bool isAudio = key.SequenceEqual("audio"u8), isText = key.SequenceEqual("text"u8), isLabels = key.SequenceEqual("labels"u8),
                    isStarts = key.SequenceEqual("starts"u8), isDurations = key.SequenceEqual("durations"u8), isSteps = key.SequenceEqual("steps"u8),
                    isSeconds = key.SequenceEqual("step_seconds"u8), isScore = key.SequenceEqual("log_probability"u8);
                json.Read();
                if (isAudio)
                {
                    audio = json.GetString();
                }
                else if (isText)
                {
                    text = json.GetString();
                }
                else if (isLabels || isStarts || isDurations)
                {
                    var values = Integers(ref json);
                    if (isLabels)
                    {
                        labels = values;
                    }
                    else if (isStarts)
                    {
                        starts = values;
                    }
                    else
                    {
                        durations = values;
                    }
                }
                else if (isSteps)
                {
                    steps = json.GetInt32();
                }
                else if (isSeconds)
                {
                    stepSeconds = json.GetDouble();
                }
                else if (isScore)
                {
                    logProbability = json.GetSingle();
                }
                else
                {
                    json.Skip();
                }
            }

            if (audio is null || text is null || labels is null || starts is null || durations is null || steps < 0)
            {
                throw new InvalidDataException("a row holds \"audio\", \"text\", \"labels\", \"starts\", \"durations\" and \"steps\"");
            }

            if (starts.Length != labels.Length || durations.Length != labels.Length)
            {
                throw new InvalidDataException($"{labels.Length} labels with {starts.Length} starts and {durations.Length} durations");
            }

            for (int i = 0; i < labels.Length; i++)
            {
                if (starts[i] < 0 || durations[i] < 0 || starts[i] + durations[i] > steps || i > 0 && starts[i] < starts[i - 1] + durations[i - 1])
                {
                    throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture, $"label {i} at step {starts[i]} for {durations[i]} steps does not follow the one before within the {steps} steps"));
                }
            }

            return new SpeechAlignment(audio, text, labels, starts, durations, steps, stepSeconds, logProbability);
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException or InvalidDataException)
        {
            throw new InvalidDataException($"{path}, line {line}: {e.Message}.", e);
        }
    }

    private static int[] Integers(ref Utf8JsonReader json)
    {
        if (json.TokenType != JsonTokenType.StartArray)
        {
            throw new InvalidDataException("labels, starts and durations are arrays of integers");
        }

        var values = new List<int>();
        while (json.Read() && json.TokenType != JsonTokenType.EndArray)
        {
            values.Add(json.GetInt32());
        }

        return [.. values];
    }

    /// <summary>
    /// A lookup from an utterance to its alignment: by its audio path exactly, else by its file name and text when one
    /// alignment has them (a source moved to another folder).
    /// </summary>
    internal static Func<SpeechUtterance, SpeechAlignment?> Index(IReadOnlyList<SpeechAlignment> alignments)
    {
        var byAudio = new Dictionary<string, SpeechAlignment>(StringComparer.Ordinal);
        var byName = new Dictionary<(string, string), SpeechAlignment?>();
        foreach (var a in alignments)
        {
            byAudio.TryAdd(a.Audio, a);
            var key = (FileName(a.Audio), a.Text);
            byName[key] = byName.ContainsKey(key) ? null : a;                    // two alike: neither is taken by name
        }

        return utterance => byAudio.TryGetValue(utterance.Audio, out var exact) ? exact
            : byName.TryGetValue((FileName(utterance.Audio), utterance.Text), out var named) ? named : null;
    }

    private static string FileName(string audio)
    {
        int bang = audio.LastIndexOf('!');
        string entry = bang >= 0 ? audio[(bang + 1)..] : audio;
        int slash = entry.LastIndexOfAny(['/', '\\']);
        return slash >= 0 ? entry[(slash + 1)..] : entry;
    }

    // One utterance's features [frames, mels] on the host: decoded, one channel, the extractor's rate.
    private static (float[] Values, int Frames) Features(ISpeechReader reader, int utterance, IAudioFeatureExtractor extractor)
    {
        var audio = AudioCodecs.Decode(reader.ReadBytes(utterance));
        if (audio.Channels > 1)
        {
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            audio = new AudioData(mono, 1, audio.SampleRate);
        }

        audio = Resampler.Resample(audio, extractor.Options.SampleRate);
        if (audio.Length < extractor.Options.WindowSize)
        {
            var padded = new float[extractor.Options.WindowSize];
            audio.Channel(0).CopyTo(padded);
            audio = new AudioData(padded, 1, audio.SampleRate);
        }

        using var extracted = extractor.Extract(audio, Device.Cpu);
        return (extracted.ToArray(), extracted.Shape[0]);
    }
}

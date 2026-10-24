// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Audio processors in the order they run, each a registered <see cref="IAudioProcessor"/>, as images have
/// <c>ImageTransformPipeline</c>. Written as text, steps comma-separated, each a name, an optional value and
/// space-separated options: <c>highpass=80, denoise=1.5 quantile=0.2, trim, normalize=-20 peak=0.9</c>; an item whose
/// name is an option of the step before it (<c>trim, silence=30</c>) belongs to that step. As JSON: that text, or an
/// array whose items are such texts or objects <c>{"name": "normalize", "value": -20, "peak": 0.9}</c>. An unknown name
/// or option, or a value out of range, is an <see cref="ArgumentException"/> naming what is taken. Immutable; equal when
/// their text is.
/// <para>
/// <see cref="Apply"/> runs a recording; <see cref="Open"/> gives one stage for blocks (a file read in pieces, a
/// stream); <see cref="Wrap"/> cleans live sound (an <see cref="IAudioInput"/>) before it is used, and refuses a
/// pipeline with a whole-recording step. The registered processors at the time of the call run.
/// </para>
/// </summary>
public sealed class AudioPipeline : IEquatable<AudioPipeline>
{
    /// <summary>The steps <paramref name="steps"/>, checked against the registry.</summary>
    /// <exception cref="ArgumentException">A step names no registered processor, or one that refuses its value or options.</exception>
    public AudioPipeline(IEnumerable<AudioProcessorStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);
        Steps = [.. steps];
        foreach (var step in Steps)
        {
            var processor = AudioProcessors.Get(step.Name);
            step.ThrowIfUnknownOptions(processor.Keys);
            processor.Check(step);
        }
    }

    /// <summary>No steps: the sound as it is.</summary>
    public static AudioPipeline Empty { get; } = new([]);

    /// <summary>The steps, in order.</summary>
    public IReadOnlyList<AudioProcessorStep> Steps { get; }

    /// <summary>Whether there are no steps.</summary>
    public bool IsEmpty => Steps.Count == 0;

    /// <summary>Whether every step runs on sound as it comes (<see cref="IAudioProcessor.Live"/>), so <see cref="Wrap"/> takes it.</summary>
    public bool Live => Steps.All(s => AudioProcessors.Get(s.Name).Live);

    /// <summary>The pipeline written as text (see the class); null, empty or "none" give <see cref="Empty"/>.</summary>
    /// <exception cref="ArgumentException">An unknown processor or option, a bad value, or an item that is not NAME or NAME=VALUE.</exception>
    public static AudioPipeline Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return Empty;
        }

        var steps = new List<(string Name, string? Value, List<KeyValuePair<string, string>> Options)>();
        foreach (string item in text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            string[] words = item.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var (name, value) = Split(words[0], item);
            if (AudioProcessors.Find(name) is null)
            {
                // An option of the step before it (written after a comma, as image pipelines do).
                if (steps.Count > 0 && value is not null && words.Length == 1
                    && AudioProcessors.Get(steps[^1].Name).Keys.Contains(name, StringComparer.OrdinalIgnoreCase))
                {
                    steps[^1].Options.Add(KeyValuePair.Create(name, value));
                    continue;
                }

                string after = steps.Count > 0 && AudioProcessors.Get(steps[^1].Name).Keys is { Count: > 0 } keys
                    ? $"; after {steps[^1].Name} it can also be one of its options: {string.Join(", ", keys)}" : "";
                throw new ArgumentException($"Unknown audio processor '{name}' (registered: {string.Join(", ", AudioProcessors.Names)}{after}).");
            }

            var options = new List<KeyValuePair<string, string>>();
            foreach (string word in words.AsSpan(1))
            {
                var (key, given) = Split(word, item);
                options.Add(KeyValuePair.Create(key, given ?? throw new ArgumentException($"An option of {name} is KEY=VALUE (such as quantile=0.2), not '{word}'.")));
            }

            steps.Add((name, value, options));
        }

        return new AudioPipeline(steps.Select(s => new AudioProcessorStep(s.Name, s.Value, s.Options)));

        static (string Name, string? Value) Split(string word, string item)
        {
            int at = word.IndexOf('=', StringComparison.Ordinal);
            string name = (at < 0 ? word : word[..at]).ToLowerInvariant();
            if (name.Length == 0)
            {
                throw new ArgumentException($"An audio processor is NAME or NAME=VALUE with KEY=VALUE options (such as highpass=80), not '{item}'.");
            }

            return (name, at < 0 ? null : word[(at + 1)..]);
        }
    }

    /// <summary>
    /// The pipeline in JSON: null, a string (as <see cref="Parse"/>), or an array of strings and objects
    /// (<c>{"name", "value", ...options}</c>).
    /// </summary>
    /// <exception cref="ArgumentException">An unknown processor or option, a bad value, or JSON of another shape.</exception>
    public static AudioPipeline FromJson(JsonNode? json)
    {
        switch (json)
        {
            case null:
                return Empty;
            case JsonValue value when value.GetValueKind() == JsonValueKind.String:
                return Parse(value.GetValue<string>());
            case JsonArray array:
            {
                var steps = new List<AudioProcessorStep>();
                foreach (var item in array)
                {
                    if (item is JsonValue text && text.GetValueKind() == JsonValueKind.String)
                    {
                        steps.AddRange(Parse(text.GetValue<string>()).Steps);
                    }
                    else if (item is JsonObject step)
                    {
                        string name = step["name"] is JsonValue n && n.GetValueKind() == JsonValueKind.String ? n.GetValue<string>()
                            : throw new ArgumentException($"An audio processor object needs a \"name\": {step.ToJsonString()}.");
                        _ = AudioProcessors.Get(name);
                        var options = step.Where(p => p.Key is not "name" and not "value" && p.Value is not null)
                            .Select(p => KeyValuePair.Create(p.Key, Text(p.Value!, p.Key))).ToList();
                        steps.Add(new AudioProcessorStep(name, step["value"] is { } v ? Text(v, "value") : null, options));
                    }
                    else
                    {
                        throw new ArgumentException($"An audio processor is a string or an object, not {item?.ToJsonString() ?? "null"}.");
                    }
                }

                return new AudioPipeline(steps);
            }

            default:
                throw new ArgumentException($"An audio pipeline is a string or an array, not {json.ToJsonString()}.");
        }

        static string Text(JsonNode node, string key) => node.GetValueKind() switch
        {
            JsonValueKind.String => node.GetValue<string>(),
            JsonValueKind.Number => node.ToJsonString(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            var kind => throw new ArgumentException($"The audio processor's '{key}' is {kind.ToString().ToLowerInvariant()}: give a string, a number or true/false."),
        };
    }

    /// <summary>This pipeline's steps, then <paramref name="next"/>'s.</summary>
    public AudioPipeline Then(AudioPipeline? next) =>
        next is null || next.IsEmpty ? this : IsEmpty ? next : new AudioPipeline(Steps.Concat(next.Steps));

    /// <summary>Whether a step is the processor <paramref name="name"/>.</summary>
    public bool Contains(string name) => Steps.Any(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));

    /// <summary>Throws unless every step is live, naming the first that is not and its live alternative when it has one.</summary>
    /// <exception cref="ArgumentException">A step needs the whole recording.</exception>
    public void ThrowIfNotLive()
    {
        foreach (var step in Steps)
        {
            var processor = AudioProcessors.Get(step.Name);
            if (!processor.Live)
            {
                throw new ArgumentException($"The audio processor {step.Name} needs the whole recording, so it cannot clean live sound"
                    + (processor.LiveAlternative is { } live ? $"; use {live} instead." : "; leave it out of a live pipeline."));
            }
        }
    }

    /// <summary>
    /// The recording after every step, in order, aligned with it: the stages' look-ahead (<see cref="IAudioStage.Latency"/>)
    /// is taken off, so a pipeline that keeps the length keeps it exactly. <paramref name="audio"/> is not changed (the
    /// result is <paramref name="audio"/> itself when no step changes it). Consecutive live steps run as one chain of
    /// stages over one copy of the samples, written in place; a whole-recording step runs its
    /// <see cref="IAudioProcessor.Apply"/>. The blocks of <see cref="Open"/> give the same samples, shifted by its
    /// <see cref="IAudioStage.Latency"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A step cannot run at the recording's rate or channels.</exception>
    public AudioData Apply(AudioData audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var current = audio;
        for (int i = 0; i < Steps.Count;)
        {
            var processor = AudioProcessors.Get(Steps[i].Name);
            if (!processor.Live)
            {
                current = processor.Apply(current, Steps[i]);
                i++;
                continue;
            }

            int end = i + 1;
            while (end < Steps.Count && AudioProcessors.Get(Steps[end].Name).Live)
            {
                end++;
            }

            using (var stage = new PipelineStage(Steps.Skip(i).Take(end - i), current.SampleRate, current.Channels))
            {
                // The samples are written in place: copied once while they are still the caller's.
                var owned = ReferenceEquals(current.Samples, audio.Samples) ? AudioStages.Copy(current) : current;
                current = AudioStages.Run(stage, owned, stage.Latency);
            }

            i = end;
        }

        return current;
    }

    /// <summary>
    /// One stage running every step over blocks of <paramref name="channels"/> channels at <paramref name="sampleRate"/>,
    /// each step opened on the rate and channels the one before it gives. Blocks are written in place (see
    /// <see cref="IAudioStage.Process"/>); a whole-recording step holds the sound until <see cref="IAudioStage.Flush"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A step cannot run at that rate or channel count.</exception>
    public IAudioStage Open(int sampleRate, int channels)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        return new PipelineStage(Steps, sampleRate, channels);
    }

    /// <summary>
    /// <paramref name="input"/> with every block run through the pipeline as it comes (its blocks are written in place),
    /// at the rate and channels the last step gives; what the stages hold back is given when the input ends. Disposing
    /// it disposes <paramref name="input"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A step needs the whole recording (<see cref="ThrowIfNotLive"/>), or cannot run at the input's rate or channels.</exception>
    public IAudioInput Wrap(IAudioInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        ThrowIfNotLive();
        return IsEmpty ? input : new ProcessedInput(input, this, Open(input.SampleRate, input.Channels));
    }

    /// <summary>The pipeline as JSON text (a string; null when empty).</summary>
    public JsonNode? ToJson() => IsEmpty ? null : JsonValue.Create(ToString());

    /// <summary>The steps written as text, ", " between them; "" when empty.</summary>
    public override string ToString() => string.Join(", ", Steps);

    /// <inheritdoc />
    public bool Equals(AudioPipeline? other) => other is not null && other.ToString() == ToString();

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as AudioPipeline);

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(ToString());

    // Steps opened one after the other, each on what the one before gives. A block goes through every stage; once one
    // gives nothing (a whole-recording stage holding the sound) the rest are not called. Flushing passes each stage's
    // tail through the stages after it, then flushes those. A whole-recording stage is given its input aligned: the
    // shift of the live stages before it (their Latency, added in seconds) is dropped from the front, as Apply drops
    // it, so the stage reads the very samples Apply gives it; the pipeline's Latency is the shift after the last one.
    private sealed class PipelineStage : IAudioStage
    {
        private readonly IAudioStage[] _stages;
        private readonly int[] _drop;                                           // samples still to drop from the front of stage i's input
        private readonly int _inputRate, _inputChannels;
        private readonly AudioData _empty;

        public PipelineStage(IEnumerable<AudioProcessorStep> steps, int sampleRate, int channels)
        {
            (_inputRate, _inputChannels) = (sampleRate, channels);
            var stages = new List<IAudioStage>();
            var drops = new List<int>();
            try
            {
                double shift = 0, delay = 0;
                foreach (var step in steps)
                {
                    var processor = AudioProcessors.Get(step.Name);
                    if (!processor.Live)
                    {
                        drops.Add((int)Math.Round(shift * sampleRate));
                        shift = 0;
                    }
                    else
                    {
                        drops.Add(0);
                    }

                    var stage = processor.Open(step, sampleRate, channels);
                    stages.Add(stage);
                    (sampleRate, channels) = (stage.SampleRate, stage.Channels);
                    shift += (double)stage.Latency / sampleRate;
                    delay += (double)stage.Delay / sampleRate;
                }

                Latency = (int)Math.Round(shift * sampleRate);
                Delay = Math.Max(Latency, (int)Math.Round(delay * sampleRate));
            }
            catch
            {
                stages.ForEach(s => s.Dispose());
                throw;
            }

            _stages = [.. stages];
            _drop = [.. drops];
            (SampleRate, Channels) = (sampleRate, channels);
            _empty = new AudioData([], channels, sampleRate);
        }

        public int SampleRate { get; }

        public int Channels { get; }

        public int Latency { get; }

        public int Delay { get; }

        public AudioData Process(AudioData block)
        {
            ArgumentNullException.ThrowIfNull(block);
            if (block.SampleRate != _inputRate || block.Channels != _inputChannels)
            {
                throw new ArgumentException($"The pipeline was opened for {_inputChannels} channel(s) at {_inputRate} Hz, not {block.Channels} at {block.SampleRate} Hz.", nameof(block));
            }

            for (int i = 0; i < _stages.Length && block.Length > 0; i++)
            {
                block = Into(i, block);
            }

            return block.Length > 0 ? block : _empty;
        }

        public AudioData Flush()
        {
            var tail = _empty;
            for (int i = 0; i < _stages.Length; i++)
            {
                var passed = tail.Length > 0 ? Into(i, tail) : tail;
                tail = AudioStages.Concat(passed, _stages[i].Flush());
            }

            return tail.Length > 0 ? tail : _empty;
        }

        // The block through stage i, the front it still drops taken off first (a whole-recording stage's aligned input).
        private AudioData Into(int i, AudioData block)
        {
            if (_drop[i] > 0)
            {
                int cut = Math.Min(_drop[i], block.Length);
                _drop[i] -= cut;
                if (cut == block.Length)
                {
                    return new AudioData([], block.Channels, block.SampleRate);
                }

                block = block.Slice(cut, block.Length - cut);
            }

            return _stages[i].Process(block);
        }

        public void Dispose()
        {
            foreach (var stage in _stages)
            {
                stage.Dispose();
            }
        }
    }

    private sealed class ProcessedInput(IAudioInput input, AudioPipeline pipeline, IAudioStage stage) : IAudioInput
    {
        public string Description { get; } = $"{input.Description}, cleaned by {pipeline}";

        public int SampleRate => stage.SampleRate;

        public int Channels => stage.Channels;

        public async IAsyncEnumerable<AudioData> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var block in input.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                var cleaned = stage.Process(block);
                if (cleaned.Length > 0)
                {
                    yield return cleaned;
                }
            }

            var tail = stage.Flush();
            if (tail.Length > 0)
            {
                yield return tail;
            }
        }

        public void Dispose()
        {
            stage.Dispose();
            input.Dispose();
        }
    }
}

/// <summary>The block plumbing the pipeline and the processors' defaults share.</summary>
internal static class AudioStages
{
    // A copy of the samples, for a stage that writes in place.
    public static AudioData Copy(AudioData audio) => new(audio.Samples.AsSpan().ToArray(), audio.Channels, audio.SampleRate);

    // A whole recording (the stage's to write) through a stage and its flush, its first `skip` samples (the stage's
    // shift) left out.
    public static AudioData Run(IAudioStage stage, AudioData owned, int skip) => Concat(stage.Process(owned), stage.Flush(), skip);

    // a then b (the same rate and channels), planar, the first `skip` samples left out; one copy at most (none when one
    // is empty and nothing is skipped).
    public static AudioData Concat(AudioData a, AudioData b, int skip = 0)
    {
        if (skip == 0 && b.Length == 0)
        {
            return a;
        }

        if (skip == 0 && a.Length == 0)
        {
            return b;
        }

        int channels = a.Channels, fromA = Math.Min(skip, a.Length), fromB = Math.Min(skip - fromA, b.Length);
        int restA = a.Length - fromA, restB = b.Length - fromB, length = restA + restB;
        var samples = GC.AllocateUninitializedArray<float>(channels * length);
        for (int c = 0; c < channels; c++)
        {
            a.Channel(c)[fromA..].CopyTo(samples.AsSpan(c * length, restA));
            b.Channel(c)[fromB..].CopyTo(samples.AsSpan(c * length + restA, restB));
        }

        return new AudioData(samples, channels, a.SampleRate);
    }

    // Blocks of the same rate and channels, in order, as one recording.
    public static AudioData Join(IReadOnlyList<AudioData> blocks, int channels, int sampleRate)
    {
        if (blocks.Count == 1)
        {
            return blocks[0];
        }

        int length = 0;
        foreach (var block in blocks)
        {
            length += block.Length;
        }

        var samples = GC.AllocateUninitializedArray<float>(channels * length);
        for (int c = 0; c < channels; c++)
        {
            int at = c * length;
            foreach (var block in blocks)
            {
                block.Channel(c).CopyTo(samples.AsSpan(at, block.Length));
                at += block.Length;
            }
        }

        return new AudioData(samples, channels, sampleRate);
    }
}

// A whole-recording processor's stage (IAudioProcessor.Open's default): the blocks are kept (they are the stage's) and
// the processor's Apply runs over them, joined, when flushed. It gives the rate and channels it was opened for (a
// whole-recording processor that changes them implements Open).
internal sealed class BufferedStage(IAudioProcessor processor, AudioProcessorStep step, int sampleRate, int channels) : IAudioStage
{
    private readonly List<AudioData> _blocks = [];
    private readonly AudioData _empty = new([], channels, sampleRate);

    public int SampleRate => sampleRate;

    public int Channels => channels;

    public AudioData Process(AudioData block)
    {
        ArgumentNullException.ThrowIfNull(block);
        if (block.SampleRate != sampleRate || block.Channels != channels)
        {
            throw new ArgumentException($"{processor.Name} was opened for {channels} channel(s) at {sampleRate} Hz, not {block.Channels} at {block.SampleRate} Hz.", nameof(block));
        }

        if (block.Length > 0)
        {
            _blocks.Add(block);
        }

        return _empty;
    }

    public AudioData Flush()
    {
        if (_blocks.Count == 0)
        {
            return _empty;
        }

        var whole = AudioStages.Join(_blocks, channels, sampleRate);
        _blocks.Clear();
        return processor.Apply(whole, step);
    }
}

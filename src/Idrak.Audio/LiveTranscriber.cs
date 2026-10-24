// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace Idrak.Audio;

/// <summary>
/// Live transcription over any speech recognizer (plan 30): the sound since the last finished segment is kept, and every
/// <see cref="LiveTranscriptionOptions.StepSeconds"/> of new sound it is transcribed again whole; the commit policy
/// (<see cref="LiveCommitPolicies"/>) says which leading words are final. A segment ends at
/// <see cref="LiveTranscriptionOptions.SilenceSeconds"/> of quiet after speech, at
/// <see cref="LiveTranscriptionOptions.WindowSeconds"/>, or at the end of the sound; its whole text is committed then and
/// carried to the next as its prompt. Sound with no speech in it is never transcribed (a recognizer invents text for
/// silence). The input is read on a task of its own while the recognizer works, so none is lost when a step takes longer
/// than the step.
/// </summary>
public static class LiveTranscriber
{
    private const int FramesPerSecond = 50;                                  // the quiet is measured per 20 ms

    /// <summary>
    /// The updates of <paramref name="sound"/>'s transcription as it arrives (any rate and channels: mixed to one, and
    /// resampled to the first block's rate when a block comes at another); ends after the last segment when the sound ends.
    /// Cancelling <paramref name="cancellationToken"/> stops at once (stop the input instead to have its last segment
    /// finished).
    /// </summary>
    /// <exception cref="ArgumentException">An option is out of range.</exception>
    /// <exception cref="NotSupportedException">The policy is not registered.</exception>
    public static async IAsyncEnumerable<LiveTranscript> TranscribeAsync(ISpeechRecognizer recognizer, IAsyncEnumerable<AudioData> sound, LiveTranscriptionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recognizer);
        ArgumentNullException.ThrowIfNull(sound);
        options ??= new LiveTranscriptionOptions();
        options.Check();
        var policy = LiveCommitPolicies.Get(options.Policy);
        var arriving = Channel.CreateUnbounded<AudioData>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var pump = Task.Run(async () =>
        {
            try
            {
                await foreach (var block in sound.WithCancellation(stop.Token).ConfigureAwait(false))
                {
                    arriving.Writer.TryWrite(block);
                }

                arriving.Writer.TryComplete();
            }
            catch (Exception e)
            {
                arriving.Writer.TryComplete(e is OperationCanceledException && stop.IsCancellationRequested ? null : e);
            }
        }, CancellationToken.None);

        var state = new Segmenter(options);
        try
        {
            while (true)
            {
                // A step's sound: everything waiting (KeepUp), or blocks until a step's worth (a recording at its own pace).
                bool ended = false;
                while (true)
                {
                    if ((options.KeepUp || state.NewSeconds < options.StepSeconds) && arriving.Reader.TryRead(out var block))
                    {
                        state.Add(block);
                        continue;
                    }

                    if (state.NewSeconds >= options.StepSeconds)
                    {
                        break;
                    }

                    if (!await arriving.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
                    {
                        ended = true;
                        break;
                    }
                }

                if (state.Step(recognizer, policy, ended, arriving.Reader, cancellationToken) is { } update)
                {
                    yield return update;
                }

                if (ended)
                {
                    break;
                }
            }
        }
        finally
        {
            stop.Cancel();
            await pump.ConfigureAwait(false);
        }
    }

    // The open segment: its mono samples, each 20 ms frame's level, where it starts in the sound, its transcriptions so far.
    private sealed class Segmenter(LiveTranscriptionOptions options)
    {
        private readonly List<float> _samples = [];
        private readonly List<double> _levels = [];                       // decibels of full scale per whole frame
        private readonly List<IReadOnlyList<string>> _heard = [];
        private float[] _mono = [];
        private int _rate;
        private int _frame;
        private double _blockSeconds;
        private int _processed;
        private int _committed;
        private double _start;                                            // seconds from the sound's start
        private double _loudest = double.NegativeInfinity;
        private string _context = options.Transcription.Prompt ?? "";

        public double NewSeconds => _rate == 0 ? 0 : (double)(_samples.Count - _processed) / _rate;

        public void Add(AudioData block)
        {
            if (block.Length == 0)
            {
                return;
            }

            if (_rate == 0)
            {
                _rate = block.SampleRate;
                _frame = Math.Max(1, _rate / FramesPerSecond);
            }
            else if (block.SampleRate != _rate)
            {
                block = Resampler.Resample(block, _rate);
            }

            _blockSeconds = (double)block.Length / _rate;
            if (_mono.Length < block.Length)
            {
                _mono = new float[block.Length];
            }

            var mono = _mono.AsSpan(0, block.Length);
            block.ToMono(mono);
            foreach (float x in mono)
            {
                _samples.Add(x);
            }

            while ((_levels.Count + 1) * _frame <= _samples.Count)
            {
                double energy = 0;
                for (int i = _levels.Count * _frame, end = i + _frame; i < end; i++)
                {
                    energy += (double)_samples[i] * _samples[i];
                }

                double level = 10 * Math.Log10(Math.Max(energy / _frame, 1e-12));
                _levels.Add(level);
                _loudest = Math.Max(_loudest, level);
            }
        }

        public LiveTranscript? Step(ISpeechRecognizer recognizer, ILiveCommitPolicy policy, bool ended, ChannelReader<AudioData> arriving, CancellationToken cancellationToken)
        {
            _processed = _samples.Count;
            if (_rate == 0)
            {
                return null;
            }

            double threshold = Math.Max(_loudest - options.SilenceDecibels, options.SilenceFloorDecibels);
            int first = _levels.FindIndex(l => l >= threshold);
            if (first < 0)
            {
                Drop(_samples.Count - (int)(options.PreRollSeconds * _rate));   // only quiet: kept as the next speech's pre-roll
                return null;
            }

            int keep = (int)Math.Round(options.PreRollSeconds * FramesPerSecond);
            if (first > keep && _committed == 0 && _heard.Count == 0)
            {
                Drop((first - keep) * _frame);                                // the quiet before the speech, but its pre-roll
            }

            int quiet = 0;
            for (int i = _levels.Count - 1; i >= 0 && _levels[i] < threshold; i--)
            {
                quiet++;
            }

            double seconds = (double)_samples.Count / _rate;
            bool finish = ended || seconds >= options.WindowSeconds || (double)quiet / FramesPerSecond >= options.SilenceSeconds;
            var clock = Stopwatch.StartNew();
            var request = options.Transcription with { Prompt = _context.Length > 0 ? _context : null };
            string text = recognizer.Transcribe(new AudioData([.. _samples], 1, _rate), request, null, cancellationToken).Text.Trim();
            var elapsed = clock.Elapsed;
            string[] words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            double start = _start, end = _start + seconds;
            double behind = Behind(arriving);
            if (finish)
            {
                string rest = string.Join(' ', words.Skip(_committed));
                Drop(_samples.Count);
                _heard.Clear();
                _committed = 0;
                if (text.Length == 0)
                {
                    return null;                                               // speech the model wrote nothing for
                }

                if (options.PromptCharacters > 0)
                {
                    string carried = _context.Length > 0 ? _context + " " + text : text;
                    _context = carried.Length > options.PromptCharacters ? carried[^options.PromptCharacters..] : carried;
                }

                return new LiveTranscript(rest, "", text, start, end, elapsed, behind);
            }

            _heard.Add(words);
            if (_heard.Count > Math.Max(1, options.Agreement))
            {
                _heard.RemoveAt(0);
            }

            int stable = Math.Clamp(policy.Stable(_heard, options), 0, words.Length);
            string committed = stable > _committed ? string.Join(' ', words[_committed..stable]) : "";
            _committed = Math.Max(_committed, stable);
            return new LiveTranscript(committed, string.Join(' ', words.Skip(_committed)), null, start, end, elapsed, behind);
        }

        // The sound waiting in the channel, in seconds: its blocks counted (not read), each as long as the last one heard.
        private double Behind(ChannelReader<AudioData> arriving) => arriving.CanCount ? arriving.Count * _blockSeconds : 0;

        // Drops the first `count` samples (whole frames) of the segment, moving its start on.
        private void Drop(int count)
        {
            count = Math.Clamp(count / _frame * _frame, 0, _samples.Count / _frame * _frame);
            if (count <= 0)
            {
                return;
            }

            _samples.RemoveRange(0, count);
            _levels.RemoveRange(0, Math.Min(_levels.Count, count / _frame));
            _processed = Math.Max(0, _processed - count);
            _start += (double)count / _rate;
        }
    }
}

/// <summary>
/// The "agreement" commit policy (<see cref="LiveCommitPolicies.Agreement"/>): the leading words the last
/// <see cref="LiveTranscriptionOptions.Agreement"/> transcriptions share (ordinal), none until there are that many.
/// </summary>
public sealed class AgreementPolicy : ILiveCommitPolicy
{
    /// <inheritdoc />
    public string Name => LiveCommitPolicies.Agreement;

    /// <inheritdoc />
    public int Stable(IReadOnlyList<IReadOnlyList<string>> transcriptions, LiveTranscriptionOptions options)
    {
        ArgumentNullException.ThrowIfNull(transcriptions);
        ArgumentNullException.ThrowIfNull(options);
        int n = options.Agreement;
        if (transcriptions.Count < n || n < 1)
        {
            return 0;
        }

        var last = transcriptions[^1];
        int shared = last.Count;
        for (int t = transcriptions.Count - n; t < transcriptions.Count - 1; t++)
        {
            var other = transcriptions[t];
            int i = 0;
            while (i < shared && i < other.Count && string.Equals(other[i], last[i], StringComparison.Ordinal))
            {
                i++;
            }

            shared = i;
        }

        return shared;
    }
}

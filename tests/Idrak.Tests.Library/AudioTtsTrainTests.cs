// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.IdrakAudio;

// Plan 21, phase S6: length regulation and an alignment's frame durations; the spec of the tone synthesizer the CLI tests build.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioTtsTrainGroup =
    [
        ("audio tts train: LengthRegulation repeats each label's row for its frames exactly (zero durations skipped, the padding after a text's frames its first row), refuses a negative duration; Durations rounds exp(x) - offset over the rate (2 halves them), never below 0, one frame when all round to none; SpeechAlignment.FrameDurations sums to the frames, gives each label the steps to the next label's start (the leading ones to the first) scaled to the frames", AudioTtsTrainRegulation),
    ];

    internal static JsonObject ToneTtsSpec() => new()
    {
        ["architecture"] = IdrakAudioFamily.Architecture, ["task"] = "tts", ["mels"] = 80, ["dim"] = 32, ["layers"] = 2, ["heads"] = 2, ["ffn_dim"] = 64,
        ["decoder_layers"] = 2, ["decoder_dim"] = 32, ["decoder_heads"] = 2, ["decoder_ffn_dim"] = 64, ["duration_filters"] = 32,
    };

    internal static void AudioTtsTrainRegulation(Device device)
    {
        // A gather: each frame is its label's row exactly.
        int[][] durations = [[2, 0, 3], [1, 4]];
        const int Length = 3, Width = 5;
        var values = new float[2 * Length * Width];
        var random = new Random(5);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var (rows, lengths) = LengthRegulation.Rows(durations, Length);
        Check(lengths.SequenceEqual([5, 5]) && rows.SequenceEqual([0, 0, 2, 2, 2, 3, 4, 4, 4, 4]), $"rows {string.Join(" ", rows)} for frames {string.Join(" ", lengths)}");
        var (padRows, padLengths) = LengthRegulation.Rows([[1, 1], [2, 0, 1]], Length);
        Check(padLengths.SequenceEqual([2, 3]) && padRows.SequenceEqual([0, 1, 0, 3, 3, 5]), $"the shorter text's padding reads its first row: {string.Join(" ", padRows)}");
        using (var scope = new TensorScope())
        {
            var (expanded, frames) = LengthRegulation.Expand(Tensor.From(values, [2, Length, Width], device), durations);
            Check(expanded.Shape.SequenceEqual([2, 5, Width]) && frames.SequenceEqual([5, 5]), $"expanded to {Tensor.FormatShape(expanded.Shape)}");
            var got = expanded.ToArray();
            for (int f = 0; f < rows.Length; f++)
            {
                Check(got.AsSpan(f * Width, Width).SequenceEqual(values.AsSpan(rows[f] * Width, Width)), $"frame {f} is row {rows[f]} exactly");
            }
        }

        ThrowsNaming<ArgumentException>(() => LengthRegulation.Rows([[1, -1]], 2), "a negative duration", "negative duration");
        ThrowsNaming<ArgumentException>(() => LengthRegulation.Rows([[1, 1, 1]], 2), "more durations than positions", "3 durations for 2 positions");

        float[] logs = [MathF.Log(5 + 1), MathF.Log(0.4f + 1), MathF.Log(10.6f + 1), MathF.Log(0.2f)];
        Check(LengthRegulation.Durations(logs, 1f).SequenceEqual([5, 0, 11, 0]), $"durations {string.Join(" ", LengthRegulation.Durations(logs, 1f))}");
        Check(LengthRegulation.Durations(logs, 1f, rate: 2f).SequenceEqual([3, 0, 5, 0]), $"at rate 2: {string.Join(" ", LengthRegulation.Durations(logs, 1f, 2f))}");
        Check(LengthRegulation.Durations([-3f, -1f, -2f], 1f).SequenceEqual([0, 1, 0]), "all none: the label predicted longest gets a frame");
        ThrowsNaming<ArgumentOutOfRangeException>(() => LengthRegulation.Durations(logs, 1f, 0f), "rate 0", "speaking rate is positive");

        // Three labels at steps 2, 5 and 9 of 12: the first from step 0, each to the next one's start, the last to the end.
        var alignment = new SpeechAlignment("a.wav", "abc", [1, 2, 3], [2, 5, 9], [2, 3, 1], 12, 0.04, -3f);
        Check(alignment.FrameDurations(12).SequenceEqual([5, 4, 3]), $"frames at the steps' scale: {string.Join(" ", alignment.FrameDurations(12))}");
        Check(alignment.FrameDurations(48).SequenceEqual([20, 16, 12]), $"four frames a step: {string.Join(" ", alignment.FrameDurations(48))}");
        var odd = alignment.FrameDurations(31);
        Check(odd.Sum() == 31 && odd.All(d => d >= 0), $"31 frames: {string.Join(" ", odd)}");
        Check(alignment.FrameDurations(0).All(d => d == 0), "no frames: none to any label");
    }
}

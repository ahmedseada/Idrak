// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 21, phase S6: training. The fixture comes from tools/audio/make_tiny_audio.py --sections ctc (ctc/: the IdrakAudio
// "ctc" network in torch with two weight sets, padded log-mel inputs, torch's CTC loss and gradients, three SGD steps and
// torchaudio's forced alignment of the step-1 log-probabilities).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioTrainGroup =
    [
        ("audio train: CtcAlignment.Viterbi gives torchaudio's forced_align path of each utterance of ctc/reference/logp-step1.npy exactly, its log-probability within 1e-4, each label's start and length from the path; MinimumSteps counts a step between equal labels; too few steps, a blank label and a label out of range refused", AudioTrainViterbi),
    ];

    private static string CtcData(string relative) => AudioData(Path.Combine("ctc", relative));

    private static JsonObject CtcReference => JsonNode.Parse(File.ReadAllText(CtcData("reference/loss.json")))!.AsObject();

    private static void AudioTrainViterbi(Device device)
    {
        var reference = CtcReference;
        var alignment = JsonNode.Parse(File.ReadAllText(CtcData("reference/alignment.json")))!.AsObject();
        int classes = (int)reference["classes"]!, blank = (int)reference["blank"]!;
        int[][] labels = [.. reference["labels"]!.AsArray().Select(l => l!.AsArray().Select(v => (int)v!).ToArray())];
        int[] steps = [.. reference["results"]!["model"]!["steps"]!.AsArray().Select(v => (int)v!)];
        var logp = ReadNpyFloat32(CtcData("reference/logp-step1.npy"));
        int padded = steps.Max();
        Check(logp.Length == labels.Length * padded * classes, $"logp-step1.npy holds {logp.Length} values");
        for (int n = 0; n < labels.Length; n++)
        {
            var rows = logp.AsSpan(n * padded * classes, steps[n] * classes);
            var path = CtcAlignment.Viterbi(rows, steps[n], classes, labels[n], blank);
            int[] expected = [.. alignment["alignments"]![n]!.AsArray().Select(v => (int)v!)];
            Check(path.Frames.SequenceEqual(expected), $"utterance {n}: the path {string.Join(",", path.Frames)} is not torchaudio's {string.Join(",", expected)}");
            float score = (float)alignment["path_log_probs"]![n]!;
            Check(MathF.Abs(path.LogProbability - score) <= 1e-4f * MathF.Abs(score), $"utterance {n}: the path's log-probability {path.LogProbability} is not {score}");
            float sum = 0;
            for (int t = 0; t < steps[n]; t++)
            {
                sum += rows[t * classes + path.Frames[t]];
            }

            Check(MathF.Abs(sum - path.LogProbability) <= 1e-4f * MathF.Abs(sum), $"utterance {n}: the log-probability {path.LogProbability} is not the frames' sum {sum}");
            for (int l = 0; l < labels[n].Length; l++)
            {
                Check(path.Lengths[l] > 0 && Enumerable.Range(path.Starts[l], path.Lengths[l]).All(t => path.Frames[t] == labels[n][l]),
                    $"utterance {n}, label {l}: steps {path.Starts[l]} + {path.Lengths[l]} do not read label {labels[n][l]}");
                Check(l == 0 || path.Starts[l] >= path.Starts[l - 1] + path.Lengths[l - 1], $"utterance {n}: label {l} starts inside label {l - 1}");
            }

            Check(path.Lengths.Sum() == path.Frames.Count(f => f != blank), $"utterance {n}: the labels' lengths do not cover the label steps");
        }

        // The same path from the tensor overload.
        using (var tensor = Tensor.From(logp.AsSpan(0, steps[0] * classes).ToArray(), [steps[0], classes], device))
        {
            var path = CtcAlignment.Viterbi(tensor, labels[0], blank);
            Check(path.Frames.SequenceEqual(alignment["alignments"]![0]!.AsArray().Select(v => (int)v!)), "the tensor overload gives the same path");
        }

        Check(CtcAlignment.MinimumSteps([4, 4, 4]) == 5 && CtcAlignment.MinimumSteps([1, 2, 2, 3]) == 5 && CtcAlignment.MinimumSteps([]) == 0, "MinimumSteps");
        var few = logp.AsSpan(0, 4 * classes).ToArray();
        ThrowsNaming<ArgumentException>(() => CtcAlignment.Viterbi(few, 4, classes, [4, 4, 4], blank), "four steps for three equal labels", "cannot hold", "need 5");
        ThrowsNaming<ArgumentException>(() => CtcAlignment.Viterbi(few, 4, classes, [blank, 3], blank), "a blank label", "is the blank");
        ThrowsNaming<ArgumentException>(() => CtcAlignment.Viterbi(few, 4, classes, [classes], blank), "a label out of range", $"not one of the {classes} classes");
    }
}

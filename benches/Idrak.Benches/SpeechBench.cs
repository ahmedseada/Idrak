// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.VitsSpeech;

namespace Idrak.Benches;

/// <summary>
/// A speech synthesizer's adversarial training step (the VITS plug-in): milliseconds a step, launches a step by operation,
/// output memory and peak memory, at one clip a step and at several, so the fixed cost of a step and the cost of each
/// extra clip show apart. IDRAK_BENCH_MODEL names the model folder (default: the tiny fixture under tests/, which shows
/// launches and memory, not real-size times); IDRAK_BENCH_CLIPS a folder of clips with transcripts (default: synthetic
/// clips written beside the output).
/// </summary>
internal static class SpeechBench
{
    private const string Section = "train-speech";

    public static void Run(Device device, Settings settings, Report report)
    {
        VitsSpeechPlugin.Register();
        string model = settings.Model ?? Fixture();
        string work = Path.Combine(Path.GetTempPath(), "idrak-benches", "speech");
        Directory.CreateDirectory(work);
        string clips = settings.Clips ?? SyntheticClips(Path.Combine(work, "clips"), 8, 6.0);
        double clipSeconds = AverageSeconds(clips);
        (string Label, double BatchSeconds)[] batches = [("one clip a step", clipSeconds * 1.01), ("four clips a step", clipSeconds * 4.01)];
        if (settings.Size == BenchSize.Large)
        {
            batches = [.. batches, ("eight clips a step", clipSeconds * 8.01)];
        }

        Console.WriteLine($"   model {model}; clips {clips} ({clipSeconds:F1} s each)");
        foreach (var (label, batchSeconds) in batches)
        {
            using var synthesizer = SpeechSynthesisModels.Load(model, new SpeechSynthesisLoadOptions { Device = device, Training = true });
            if (synthesizer is not IAdversarialSpeechSynthesizer adversarial)
            {
                throw new NotSupportedException($"{model} is not an adversarial synthesizer");
            }

            const int Steps = 3;                                                         // one cold, two warm
            var options = new SpeechTrainingOptions
            {
                Steps = Steps, BatchSeconds = batchSeconds, LearningRate = 2e-5f, Schedule = "constant", FeatureCache = null,
                Adversarial = new AdversarialSynthesisOptions(), SaveEvery = 0, KeepCheckpoints = 1,
            };
            var census = new Census();
            var clock = Stopwatch.StartNew();
            var progress = new StepTimes(clock);
            string output = Path.Combine(work, "run-" + label.Replace(' ', '-'));
            if (Directory.Exists(output))
            {
                Directory.Delete(output, recursive: true);
            }

            ComputeResources.ResetPeakMemoryUsage(device);
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            using (Telemetry.Subscribe(census))
            {
                AdversarialSynthesisTrainer.Train(adversarial, new SpeechSource(SpeechSources.AudioFolder, [clips]), null, options, output, null, progress);
            }

            long peak = ComputeResources.GetMemoryUsage(device).Peak - before;
            double warm = progress.Times.Count > 1 ? progress.Times.Skip(1).Min() : progress.Times.FirstOrDefault();
            double cold = progress.Times.FirstOrDefault();
            double launches = census.Count / (double)Steps;
            Console.WriteLine(Bench.Line(label, ("s warm", Bench.F(warm, "0.00")), ("s cold", Bench.F(cold, "0.00")), ("launches/step", Bench.F(launches, "0")),
                ("peak", Bench.Mib(peak)), ("x real time", Bench.F(batchSeconds / warm, "0.0"))));
            report.Add(Section, label, device, "warm", warm * 1000, "ms");
            report.Add(Section, label, device, "cold", cold * 1000, "ms");
            report.Add(Section, label, device, "launches", launches, "launches");
            report.Add(Section, label, device, "peak", peak / 1048576.0, "MiB");
            report.Add(Section, label, device, "speed", batchSeconds / warm, "x real time");
            Console.WriteLine("   launches a step by operation (the top sixteen):");
            foreach (var (operation, count, elements, time) in census.ByCount.Take(16))
            {
                Console.WriteLine(Bench.Line("   " + operation, ("/step", Bench.F(count / (double)Steps, "0")), ("MiB out", Bench.F(elements * 4.0 / 1048576 / Steps, "0")),
                    ("ms", Bench.F(time.TotalMilliseconds / Steps, "0"))));
                report.Add(Section, $"{label}: {operation}", device, "launches", count / (double)Steps, "launches");
            }
        }
    }

    // The tiny VITS fixture the training tests use, found from the build output folder.
    private static string Fixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "tests", "Idrak.Tests", "data", "audio", "vits-train", "model");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException("the VITS fixture (tests/Idrak.Tests/data/audio/vits-train/model) was not found above the build output; name a model with IDRAK_BENCH_MODEL");
    }

    // Clips of voiced noise (a buzz with vibrato under noise) with transcripts, so the trainer has utterances of a known length.
    private static string SyntheticClips(string folder, int count, double seconds)
    {
        Directory.CreateDirectory(folder);
        const int Rate = 16000;
        var random = new Random(17);
        string[] words = ["this", "voice", "speaks", "clear", "morning", "test", "model", "good", "sound", "steady"];
        for (int c = 0; c < count; c++)
        {
            string wav = Path.Combine(folder, $"clip-{c:00}.wav");
            if (File.Exists(wav))
            {
                continue;
            }

            int n = (int)(seconds * Rate);
            var samples = new float[n];
            double phase = 0, pitch = 110 + 20 * c;
            for (int i = 0; i < n; i++)
            {
                double t = i / (double)Rate;
                phase += 2 * Math.PI * (pitch + 5 * Math.Sin(2 * Math.PI * 5 * t)) / Rate;
                double voiced = 0.3 * Math.Sin(phase) + 0.15 * Math.Sin(2 * phase) + 0.08 * Math.Sin(3 * phase);
                samples[i] = (float)((voiced + 0.02 * (random.NextDouble() * 2 - 1)) * (0.6 + 0.4 * Math.Sin(2 * Math.PI * 0.7 * t)));
            }

            AudioEncoders.Save(wav, new AudioData(samples, 1, Rate));
            File.WriteAllText(Path.ChangeExtension(wav, ".txt"), string.Join(' ', Enumerable.Range(0, 12).Select(_ => words[random.Next(words.Length)])));
        }

        return folder;
    }

    private static double AverageSeconds(string folder)
    {
        var files = Directory.GetFiles(folder, "*.wav");
        if (files.Length == 0)
        {
            throw new FileNotFoundException($"{folder} holds no .wav clips");
        }

        double total = 0;
        foreach (string file in files)
        {
            var info = AudioCodecs.ReadInfo(file) ?? throw new InvalidDataException($"{file}: not a known audio file");
            total += info.Samples / (double)info.SampleRate / Math.Max(1, info.Channels);
        }

        return total / files.Length;
    }

    private sealed class StepTimes(Stopwatch clock) : IProgress<SpeechTrainingProgress>
    {
        private double _last;

        public List<double> Times { get; } = [];

        public void Report(SpeechTrainingProgress value)
        {
            if (value.Stage != "step")
            {
                return;
            }

            double now = clock.Elapsed.TotalSeconds;
            Times.Add(now - _last);
            _last = now;
        }
    }
}

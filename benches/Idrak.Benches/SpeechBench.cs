// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.VitsSpeech;

namespace Idrak.Benches;

/// <summary>
/// A speech synthesizer's adversarial training step (the VITS plug-in): seconds a step, launches a step by operation,
/// output memory and peak memory, at one clip a step and at several, so the fixed cost of a step and the cost of each
/// extra clip show apart. A run is one cold step then IDRAK_BENCH_CALLS timed ones; each step's time is listed with the
/// device tuning in it (a batch whose count or padded length lands in a new size class tunes every convolution shape
/// once, kept in the device's cache file), and the warm time is the fastest timed step with the audio it trained.
/// IDRAK_BENCH_MODEL names the model folder (default: the tiny fixture under tests/, which shows launches and memory,
/// not real-size times); IDRAK_BENCH_CLIPS a folder of clips with transcripts (default: synthetic clips written beside
/// the output).
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

            int steps = 1 + settings.Calls;                                              // one cold, then the timed ones
            var options = new SpeechTrainingOptions
            {
                Steps = steps, BatchSeconds = batchSeconds, LearningRate = 2e-5f, Schedule = "constant", FeatureCache = null,
                Adversarial = new AdversarialSynthesisOptions(), SaveEvery = 0, KeepCheckpoints = 1,
            };
            // One census for the first step (loading, the first tuning), another for the rest: the warm steps' launches and
            // host time are read without the cold step's tuning in them.
            var cold = new Census();
            var warm = new Census();
            var clock = Stopwatch.StartNew();
            var progress = new StepTimes(clock, device, cold, warm);
            string output = Path.Combine(work, "run-" + label.Replace(' ', '-'));
            if (Directory.Exists(output))
            {
                Directory.Delete(output, recursive: true);
            }

            ComputeResources.ResetPeakMemoryUsage(device);
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            SpeechTrainingResult result;
            using (progress)
            {
                result = AdversarialSynthesisTrainer.Train(adversarial, new SpeechSource(SpeechSources.AudioFolder, [clips]), null, options, output, null, progress);
            }

            long peak = ComputeResources.GetMemoryUsage(device).Peak - before;
            // The first step apart, and the last (it writes the checkpoint) when others remain: the fastest of the rest, with
            // the audio it trained, is the step's time; the list shows the steps that tuned.
            var timed = progress.Steps.Skip(1).Take(Math.Max(1, progress.Steps.Count - 2)).ToList();
            var fastest = timed.Count > 0 ? timed.MinBy(s => s.Seconds) : progress.Steps.FirstOrDefault();
            double warmSeconds = fastest.Seconds, coldSeconds = progress.Steps.FirstOrDefault().Seconds;
            int warmSteps = Math.Max(1, progress.Steps.Count - 1);
            double launches = warm.Count / (double)warmSteps;
            double speed = fastest.Audio > 0 ? fastest.Audio / warmSeconds : batchSeconds / warmSeconds;
            long tunings = cold.Tunings + warm.Tunings;
            double tuningSeconds = (cold.TuningTime + warm.TuningTime).TotalSeconds;
            Console.WriteLine(Bench.Line(label, ("s warm", Bench.F(warmSeconds, "0.00")), ("s cold", Bench.F(coldSeconds, "0.00")), ("launches/step", Bench.F(launches, "0")),
                ("peak", Bench.Mib(peak)), ("x real time", Bench.F(speed, "0.0"))));
            Console.WriteLine("   steps: " + string.Join(", ", progress.Steps.Select(s => Bench.F(s.Seconds, "0.0") + (s.Tuning >= 0.05 ? $" (tuning {Bench.F(s.Tuning, "0.0")})" : "")))
                + $" s; the fastest timed step trained {Bench.F(fastest.Audio, "0.0")} s of audio; {tunings} tunings, {Bench.F(tuningSeconds, "0.0")} s in all");
            Console.WriteLine("   tunings in the first step: " + Tunings(cold) + "; in the steps after: " + Tunings(warm));
            foreach (string note in result.Notes)
            {
                Console.WriteLine("   note: " + note);
            }

            report.Add(Section, label, device, "warm", warmSeconds * 1000, "ms");
            report.Add(Section, label, device, "cold", coldSeconds * 1000, "ms");
            report.Add(Section, label, device, "launches", launches, "launches");
            report.Add(Section, label, device, "peak", peak / 1048576.0, "MiB");
            report.Add(Section, label, device, "speed", speed, "x real time");
            report.Add(Section, label, device, "tuning", tuningSeconds * 1000, "ms");
            report.Add(Section, label, device, "host time a warm step", warm.ByTime.Sum(o => o.Time.TotalMilliseconds) / warmSteps, "ms");
            Console.WriteLine("   launches a warm step by operation (the top sixteen; ms: the host's time a step in its calls, launching and where it waited):");
            foreach (var (operation, count, elements, time) in warm.ByCount.Take(16))
            {
                Console.WriteLine(Bench.Line("   " + operation, ("/step", Bench.F(count / (double)warmSteps, "0")), ("MiB out", Bench.F(elements * 4.0 / 1048576 / warmSteps, "0")),
                    ("ms", Bench.F(time.TotalMilliseconds / warmSteps, "0"))));
                report.Add(Section, $"{label}: {operation}", device, "launches", count / (double)warmSteps, "launches");
            }

            Console.WriteLine($"   the host's time a warm step by operation (the top eight of {Bench.F(warm.ByTime.Sum(o => o.Time.TotalMilliseconds) / warmSteps, "0")} ms in calls):");
            foreach (var (operation, count, _, time) in warm.ByTime.Take(8))
            {
                Console.WriteLine(Bench.Line("   " + operation, ("ms", Bench.F(time.TotalMilliseconds / warmSteps, "0")), ("/step", Bench.F(count / (double)warmSteps, "0")),
                    ("µs a call", Bench.F(time.TotalMilliseconds * 1000 / Math.Max(1, count), "0"))));
            }
        }
    }

    // "Convolution 212 (15.2 s), TensorSplits 40 (2.1 s)" or "none".
    private static string Tunings(Census census)
    {
        var parts = census.TuningsByOperation.Select(t => $"{t.Operation} {t.Count} ({Bench.F(t.Time.TotalSeconds, "0.0")} s)").ToList();
        return parts.Count == 0 ? "none" : string.Join(", ", parts);
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

    // The steps' times (the device drained first), each with the tuning in it and the audio it trained; the cold census
    // listens through the first step, the warm one after it.
    private sealed class StepTimes(Stopwatch clock, Device device, Census cold, Census warm) : IProgress<SpeechTrainingProgress>, IDisposable
    {
        private IDisposable? _subscription = Telemetry.Subscribe(cold);
        private double _last, _lastTuning;

        /// <summary>Each step: its seconds (the device drained), the seconds of tuning in it, the audio seconds it trained.</summary>
        public List<(double Seconds, double Tuning, double Audio)> Steps { get; } = [];

        public void Report(SpeechTrainingProgress value)
        {
            if (value.Stage != "step")
            {
                return;
            }

            device.Backend.Synchronize();                                   // the step's queued work is the step's time, not the next one's
            double now = clock.Elapsed.TotalSeconds, tuning = (cold.TuningTime + warm.TuningTime).TotalSeconds;
            Steps.Add((now - _last, tuning - _lastTuning, value.BatchAudioSeconds));
            (_last, _lastTuning) = (now, tuning);
            if (Steps.Count == 1)
            {
                _subscription?.Dispose();
                _subscription = Telemetry.Subscribe(warm);
            }
        }

        public void Dispose()
        {
            _subscription?.Dispose();
            _subscription = null;
        }
    }
}

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Text.Plugins;

namespace Idrak.Benches;

/// <summary>
/// A text encoder-decoder (any family a plug-in registers in <c>PretrainedSeq2Seq</c>; IDRAK_BENCH_SEQ2SEQ names the
/// folder, default the tiny fixture under tests/): the load (cold and warm, and the host memory it allocated), a cached
/// decoding step through a context of the model's longest target (ms a token, launches a token), and a beam search of 4
/// beams that runs its full length (ms a step), with the best hypothesis's tokens hashed and its score printed so a change
/// can be shown to keep the result (rule 71).
/// </summary>
internal static class Seq2SeqBench
{
    private const string Section = "seq2seq";

    public static void Run(Device device, Settings settings, Report report)
    {
        TextPlugins.RegisterAll();
        string folder = Environment.GetEnvironmentVariable("IDRAK_BENCH_SEQ2SEQ") is { Length: > 0 } chosen ? chosen : Fixture();
        int sourceLength = 32, steps = settings.Size == BenchSize.Small ? 32 : 64, beams = 4;

        // The load: cold (the first in the process), then warm; the host bytes it allocated (all threads).
        PretrainedSeq2SeqModel Load(out double ms, out long hostBytes)
        {
            GC.Collect();
            long allocated = GC.GetTotalAllocatedBytes(precise: true);
            var watch = Stopwatch.StartNew();
            var loaded = PretrainedSeq2SeqModel.Load(folder, new PretrainedEncoderOptions { Device = device });
            device.Backend.Synchronize();
            ms = watch.Elapsed.TotalMilliseconds;
            hostBytes = GC.GetTotalAllocatedBytes(precise: true) - allocated;
            return loaded;
        }

        Load(out double coldLoad, out long coldBytes).Dispose();
        using var pretrained = Load(out double warmLoad, out long warmBytes);
        var model = pretrained.Model;
        string name = $"{pretrained.Architecture} {Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(folder)))}";
        Console.WriteLine(Bench.Line($"load {name}", ("ms cold", Bench.F(coldLoad, "0")), ("ms warm", Bench.F(warmLoad, "0")),
            ("MiB host cold", Bench.F(coldBytes / 1048576.0, "0.0")), ("MiB host warm", Bench.F(warmBytes / 1048576.0, "0.0"))));
        report.Add(Section, $"load {name}", device, "cold", coldLoad, "ms");
        report.Add(Section, $"load {name}", device, "warm", warmLoad, "ms");
        report.Add(Section, $"load {name}", device, "host allocated", warmBytes, "bytes");

        var random = new Random(3);
        var source = new float[sourceLength];
        for (int i = 0; i < sourceLength; i++)
        {
            source[i] = random.Next(2, model.Vocabulary);
        }

        int capacity = Math.Max(steps + 1, model.MaxTargetPositions - 1);
        void Encode(DecodingContext context)
        {
            model.Network.Eval();
            using (Autograd.NoGrad())
            using (new TensorScope())
            {
                model.SetMemory(model.Encoder.Encode(Tensor.From(source, [1, sourceLength], device)), context);
            }
        }

        // Cached decoding, one token a step through a context of the longest target (what the generator makes): the first
        // step apart (cold), then the median of the rounds' mean step.
        var tokens = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            tokens[i] = random.Next(2, model.Vocabulary);
        }

        double DecodeRound(out long launches, out double firstMs)
        {
            using var context = new DecodingContext(device, 1, capacity) { LastPositionOnly = true };
            Encode(context);
            launches = 0;
            firstMs = 0;
            var watch = Stopwatch.StartNew();
            using (Autograd.NoGrad())
            {
                for (int t = 0; t < steps; t++)
                {
                    var step = Stopwatch.StartNew();
                    using var scope = new TensorScope();
                    float id = t == 0 ? model.DecoderStartToken : tokens[t - 1];
                    if (t == steps - 1)
                    {
                        launches = Bench.Count(() => model.DecodeCached(Tensor.From([id], [1, 1], device), context)).Count;
                    }
                    else
                    {
                        model.DecodeCached(Tensor.From([id], [1, 1], device), context);
                    }

                    device.Backend.Synchronize();
                    if (t == 0)
                    {
                        firstMs = step.Elapsed.TotalMilliseconds;
                    }
                }
            }

            return watch.Elapsed.TotalMilliseconds / steps;
        }

        DecodeRound(out _, out double coldStep);
        var stepTimes = new double[Math.Max(1, settings.Repeats)];
        long stepLaunches = 0;
        for (int r = 0; r < stepTimes.Length; r++)
        {
            stepTimes[r] = DecodeRound(out stepLaunches, out _);
        }

        Array.Sort(stepTimes);
        double warmStep = stepTimes[stepTimes.Length / 2];
        string decodeCase = $"cached step {name}, {steps} tokens, {capacity} slots";
        Console.WriteLine(Bench.Line(decodeCase, ("ms first step cold", Bench.F(coldStep, "0.00")), ("ms/token warm", Bench.F(warmStep, "0.00")),
            ("launches/token", stepLaunches.ToString(System.Globalization.CultureInfo.InvariantCulture))));
        report.Add(Section, decodeCase, device, "first step cold", coldStep, "ms");
        report.Add(Section, decodeCase, device, "a token warm", warmStep, "ms");
        report.Add(Section, decodeCase, device, "launches a token", stepLaunches, "launches");

        // A beam search of `beams` beams for `steps` tokens (no end token: every search runs its full length).
        IReadOnlyList<BeamHypothesis> Search(out double ms)
        {
            using var context = new DecodingContext(device, beams, capacity) { LastPositionOnly = true };
            Encode(context);
            var watch = Stopwatch.StartNew();
            var found = BeamSearch.Decode((ids, ctx) => model.DecodeCached(ids, ctx), context, [model.DecoderStartToken],
                new BeamOptions { Beams = beams, MaxTokens = steps, EndToken = -1 });
            device.Backend.Synchronize();
            ms = watch.Elapsed.TotalMilliseconds;
            return found;
        }

        var first = Search(out double coldSearch);
        var searchTimes = new double[Math.Max(1, settings.Repeats)];
        for (int r = 0; r < searchTimes.Length; r++)
        {
            Search(out searchTimes[r]);
        }

        Array.Sort(searchTimes);
        double warmSearch = searchTimes[searchTimes.Length / 2];
        string beamCase = $"beam search {name}, {beams} beams, {steps} tokens";
        var best = first[0];
        ulong hash = 1469598103934665603UL;
        foreach (int token in best.Tokens)
        {
            hash = (hash ^ (uint)token) * 1099511628211UL;
        }

        Console.WriteLine(Bench.Line(beamCase, ("ms cold", Bench.F(coldSearch, "0")), ("ms warm", Bench.F(warmSearch, "0")), ("ms/step warm", Bench.F(warmSearch / steps, "0.00"))));
        Console.WriteLine($"   best of {first.Count}: {best.Tokens.Count} tokens, hash {hash:x16}, score {best.Score:R}, log-probability {best.LogProb:R}; "
                          + $"scores {string.Join(" ", first.Select(h => h.Score.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}");
        report.Add(Section, beamCase, device, "search cold", coldSearch, "ms");
        report.Add(Section, beamCase, device, "search warm", warmSearch, "ms");
    }

    // The tiny encoder-decoder fixture the plug-in tests use, found from the build output folder.
    private static string Fixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "tests", "Idrak.Tests", "data", "plugins", "t5", "tiny-t5");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException("the tiny encoder-decoder fixture (tests/Idrak.Tests/data/plugins/t5/tiny-t5) was not found above the build output; name a folder with IDRAK_BENCH_SEQ2SEQ");
    }
}
